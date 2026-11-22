import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageDraw
from ultralytics import YOLO

import io
import json
import hashlib
import zipfile
import colorsys
import gc
import uuid


# =========================================================
# AI IMAGE ANNOTATION - COMPACT WORKFLOW
# =========================================================
st.set_page_config(
    page_title="AI Image Annotation",
    page_icon="🖼️",
    layout="wide"
)

YOLO_MODEL = "yolo11n.pt"
CONFIDENCE = 0.20
MAX_DETECTIONS = 50
CANVAS_MAX_WIDTH = 850

st.title("AI Image Annotation")
st.caption("Upload → Detect & Label → Describe → Add Missing Objects → Download")

# -------------------- Session state --------------------
DEFAULTS = {
    "objects": [],
    "image_hash": None,
    "scene_description": "",
    "descriptions_generated": False,
    "manual_canvas_revision": 0,
    "pending_manual_bbox": None,
    "last_deleted": None,
}
for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# -------------------- Helpers --------------------
def uid():
    return uuid.uuid4().hex[:12]


def color_for(index):
    hue = (index * 0.61803398875) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.78, 0.95)
    return tuple(int(v * 255) for v in rgb)


def next_label(class_name, objects):
    class_name = class_name.strip().lower()
    used = set()
    for obj in objects:
        if obj.get("class_name", "").lower() == class_name:
            try:
                used.add(int(obj["label"].rsplit(" ", 1)[1]))
            except (KeyError, ValueError, IndexError):
                pass
    number = 1
    while number in used:
        number += 1
    return f"{class_name.title()} {number}"


def box_position(box, image_width, image_height):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    h = "left" if cx < image_width / 3 else "right" if cx > image_width * 2 / 3 else "center"
    v = "top" if cy < image_height / 3 else "bottom" if cy > image_height * 2 / 3 else "middle"
    if v == "middle":
        return h
    if h == "center":
        return v
    return f"{v}-{h}"


def box_size(box, image_width, image_height):
    x1, y1, x2, y2 = box
    ratio = max(0, x2 - x1) * max(0, y2 - y1) / max(1, image_width * image_height)
    return "small" if ratio < 0.025 else "medium" if ratio < 0.15 else "large"


def describe_object(index, objects, image_width, image_height):
    obj = objects[index]
    position = box_position(obj["bbox"], image_width, image_height)
    size = box_size(obj["bbox"], image_width, image_height)
    x1, y1, x2, y2 = obj["bbox"]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    diagonal = max(1, (image_width ** 2 + image_height ** 2) ** 0.5)
    nearby = []
    for j, other in enumerate(objects):
        if j == index:
            continue
        a, b, c, d = other["bbox"]
        ox, oy = (a + c) / 2, (b + d) / 2
        distance = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
        if distance / diagonal < 0.22:
            nearby.append((distance, other["label"]))
    nearby.sort(key=lambda item: item[0])

    text = (
        f"{obj['label']} is a {obj['class_name']}. "
        f"It is located in the {position} area of the image. "
        f"Its bounding box covers a {size} portion of the image."
    )
    if nearby:
        text += " Nearby annotated objects include " + ", ".join(x[1] for x in nearby[:3]) + "."
    return text


def scene_description(objects, image_width, image_height):
    if not objects:
        return "No objects have been annotated."
    counts = {}
    for obj in objects:
        name = obj["class_name"].lower()
        counts[name] = counts.get(name, 0) + 1
    summary = [
        f"{count} {name}" + ("" if count == 1 else "s")
        for name, count in counts.items()
    ]
    positions = [
        f"{obj['label']} is in the {box_position(obj['bbox'], image_width, image_height)} area"
        for obj in objects
    ]
    return (
        f"The image contains {len(objects)} annotated objects: "
        + ", ".join(summary)
        + ". Approximate positions: "
        + "; ".join(positions)
        + "."
    )


def render_labelled_image(source, objects):
    output = source.copy().convert("RGB")
    draw = ImageDraw.Draw(output)
    line_width = max(2, output.width // 500)
    for index, obj in enumerate(objects):
        x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
        color = color_for(index)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        label = obj["label"]
        try:
            bb = draw.textbbox((x1, y1), label)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
        except Exception:
            tw, th = len(label) * 8, 14
        top = max(0, y1 - th - 8)
        draw.rectangle((x1, top, x1 + tw + 10, top + th + 7), fill=color)
        draw.text((x1 + 5, top + 3), label, fill="black")
    return output


def set_annotations_changed():
    st.session_state.descriptions_generated = False
    st.session_state.scene_description = ""


# -------------------- Upload --------------------
uploaded = st.file_uploader(
    "Upload a real-world image",
    type=["jpg", "jpeg", "png", "webp"]
)
if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()

try:
    image_bytes = uploaded.getvalue()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
except Exception as exc:
    st.error(f"Could not open image: {exc}")
    st.stop()

image_width, image_height = image.size
image_hash = hashlib.sha256(image_bytes).hexdigest()

if st.session_state.image_hash != image_hash:
    st.session_state.objects = []
    st.session_state.image_hash = image_hash
    st.session_state.scene_description = ""
    st.session_state.descriptions_generated = False
    st.session_state.manual_canvas_revision += 1
    st.session_state.pending_manual_bbox = None
    st.session_state.last_deleted = None

st.caption(f"Image dimensions: {image_width} × {image_height}")

# Keep the workflow compact: three tabs instead of one long page.
tab_detect, tab_describe, tab_review = st.tabs([
    "1 · Detect & Preview",
    "2 · Descriptions & Manual Add",
    "3 · Final Review & Download"
])


# =========================================================
# TAB 1: DETECT + PREVIEW
# =========================================================
with tab_detect:
    st.image(image, caption="Uploaded image", use_container_width=True)
    st.subheader("Object Detection")
    st.write("Click Detect Objects to create bounding boxes and labels.")

    if st.button("🔍 Detect Objects", type="primary", use_container_width=True):
        model = None
        predictions = None
        try:
            with st.spinner("Detecting objects..."):
                model = YOLO(YOLO_MODEL)
                predictions = model.predict(
                    source=image,
                    imgsz=512,
                    conf=CONFIDENCE,
                    max_det=MAX_DETECTIONS,
                    device="cpu",
                    verbose=False
                )
                result = predictions[0]
                detected = []
                counts = {}

                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().tolist()
                    classes = result.boxes.cls.cpu().tolist()
                    scores = result.boxes.conf.cpu().tolist()

                    for box, class_id, score in zip(boxes, classes, scores):
                        class_name = str(result.names[int(class_id)]).lower()
                        counts[class_name] = counts.get(class_name, 0) + 1
                        x1, y1, x2, y2 = [int(round(v)) for v in box]
                        x1 = max(0, min(x1, image_width))
                        y1 = max(0, min(y1, image_height))
                        x2 = max(0, min(x2, image_width))
                        y2 = max(0, min(y2, image_height))
                        if x2 <= x1 or y2 <= y1:
                            continue

                        detected.append({
                            "id": uid(),
                            "label": f"{class_name.title()} {counts[class_name]}",
                            "class_name": class_name,
                            "confidence": float(score),
                            "bbox": [x1, y1, x2, y2],
                            "description": "",
                            "description_source": "",
                            "source": "YOLO"
                        })

                st.session_state.objects = detected
                # Generate an initial description for every detected object automatically.
                for i, obj in enumerate(st.session_state.objects):
                    obj["description"] = describe_object(
                        i, st.session_state.objects, image_width, image_height
                    )
                    obj["description_source"] = "generated"
                st.session_state.scene_description = scene_description(
                    st.session_state.objects, image_width, image_height
                )
                st.session_state.descriptions_generated = True
                st.session_state.pending_manual_bbox = None
                st.session_state.manual_canvas_revision += 1

            st.success(f"Detected {len(detected)} object(s) and generated descriptions.")
        except Exception as exc:
            st.error(f"Detection failed: {exc}")
        finally:
            if predictions is not None:
                del predictions
            if model is not None:
                del model
            gc.collect()

    st.subheader("Detected / Annotated Image")
    if st.session_state.objects:
        st.image(
            render_labelled_image(image, st.session_state.objects),
            caption="Every current object is shown with its bounding box and label",
            use_container_width=True
        )
    else:
        st.info("Click Detect Objects. Detected boxes and labels will appear here.")


# =========================================================
# TAB 2: DESCRIPTIONS + MANUAL ADD
# =========================================================
with tab_describe:
    st.subheader("Generate Descriptions")
    st.caption(
        "Instant descriptions are based on object class, box position, box size "
        "and nearby annotations. They do not verify fine visual details."
    )

    if st.button("✨ Generate AI Descriptions", use_container_width=True):
        with st.spinner("Generating instant descriptions..."):
            for i, obj in enumerate(st.session_state.objects):
                # Keep descriptions the user already wrote.
                if not obj.get("description", "").strip() or obj.get("description_source") == "generated":
                    obj["description"] = describe_object(
                        i, st.session_state.objects, image_width, image_height
                    )
                    obj["description_source"] = "generated"

            st.session_state.scene_description = scene_description(
                st.session_state.objects, image_width, image_height
            )
            st.session_state.descriptions_generated = True
        st.success("Descriptions generated.")

    if st.session_state.objects:
        with st.expander("View / Edit Individual Object Descriptions", expanded=True):
            st.caption(
                "You can add your own description or correct an incomplete generated description."
            )
            for obj in st.session_state.objects:
                st.markdown(f"**{obj['label']} · {obj['source']}**")
                with st.form(f"individual_description_{obj['id']}"):
                    edited_description = st.text_area(
                        "Object description",
                        value=obj.get("description", ""),
                        placeholder="Add or edit details about this object...",
                        key=f"individual_desc_input_{obj['id']}"
                    )
                    save_description = st.form_submit_button("Save Description")
                if save_description:
                    obj["description"] = edited_description.strip()
                    obj["description_source"] = "manual"
                    set_annotations_changed()
                    st.success(f"Description saved for {obj['label']}.")
                    st.rerun()
        if st.session_state.descriptions_generated:
            with st.expander("Complete scene description", expanded=True):
                st.write(st.session_state.scene_description)
    else:
        st.info("Detect objects first, or add a missing object manually below.")

    st.divider()
    st.subheader("Add a Missing Object Manually")
    st.write(
        "Draw a rectangle around any object YOLO missed. Enter its class and "
        "description, then add it to the same final annotations."
    )

    canvas_width = min(image_width, CANVAS_MAX_WIDTH)
    scale = canvas_width / image_width
    canvas_height = max(1, int(round(image_height * scale)))
    canvas_background = image.resize((canvas_width, canvas_height))

    canvas_key = f"manual_canvas_{image_hash}_{st.session_state.manual_canvas_revision}"

    # Re-display the last saved pending rectangle after Streamlit reruns.
    initial_drawing = None
    if st.session_state.pending_manual_bbox:
        bx1, by1, bx2, by2 = st.session_state.pending_manual_bbox
        initial_drawing = {
            "version": "4.4.0",
            "objects": [{
                "type": "rect",
                "left": bx1 * scale,
                "top": by1 * scale,
                "width": max(1, (bx2 - bx1) * scale),
                "height": max(1, (by2 - by1) * scale),
                "fill": "rgba(0, 180, 90, 0.18)",
                "stroke": "#00AA66",
                "strokeWidth": 3,
                "selectable": True,
                "evented": True
            }]
        }

    manual_canvas = st_canvas(
        fill_color="rgba(0, 180, 90, 0.18)",
        stroke_width=3,
        stroke_color="#00AA66",
        background_image=canvas_background,
        initial_drawing=initial_drawing,
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="rect",
        key=canvas_key
    )

    # Keep the most recently drawn rectangle in session state. Accept both
    # dictionary and JSON-string responses from drawable-canvas versions.
    canvas_json = getattr(manual_canvas, "json_data", None) if manual_canvas else None
    if isinstance(canvas_json, str):
        try:
            canvas_json = json.loads(canvas_json)
        except (TypeError, json.JSONDecodeError):
            canvas_json = None

    if isinstance(canvas_json, dict):
        canvas_objects = canvas_json.get("objects") or []
        rects = [
            item for item in canvas_objects
            if str(item.get("type", "")).lower() == "rect"
        ]
        if rects:
            shape = rects[-1]
            left = float(shape.get("left", 0) or 0)
            top = float(shape.get("top", 0) or 0)
            rect_w = float(shape.get("width", 0) or 0) * float(shape.get("scaleX", 1) or 1)
            rect_h = float(shape.get("height", 0) or 0) * float(shape.get("scaleY", 1) or 1)

            x1 = max(0, min(image_width, int(round(left / scale))))
            y1 = max(0, min(image_height, int(round(top / scale))))
            x2 = max(0, min(image_width, int(round((left + rect_w) / scale))))
            y2 = max(0, min(image_height, int(round((top + rect_h) / scale))))

            if x2 > x1 and y2 > y1:
                st.session_state.pending_manual_bbox = [x1, y1, x2, y2]

    pending_bbox = st.session_state.pending_manual_bbox
    if pending_bbox:
        st.success(
            f"Box ready: x1={pending_bbox[0]}, y1={pending_bbox[1]}, "
            f"x2={pending_bbox[2]}, y2={pending_bbox[3]}"
        )

        # Move and resize the pending box using its coordinates.
        box_key = "_".join(str(v) for v in pending_bbox)
        st.caption("Adjust the box position or size before saving.")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            edit_x = st.number_input(
                "Move X", min_value=0, max_value=max(0, image_width - 1),
                value=int(pending_bbox[0]), step=1,
                key=f"manual_x_{image_hash}_{box_key}"
            )
        with c2:
            edit_y = st.number_input(
                "Move Y", min_value=0, max_value=max(0, image_height - 1),
                value=int(pending_bbox[1]), step=1,
                key=f"manual_y_{image_hash}_{box_key}"
            )
        with c3:
            edit_w = st.number_input(
                "Width", min_value=1, max_value=max(1, image_width - int(edit_x)),
                value=max(1, min(int(pending_bbox[2] - pending_bbox[0]),
                                 image_width - int(edit_x))),
                step=1, key=f"manual_w_{image_hash}_{box_key}"
            )
        with c4:
            edit_h = st.number_input(
                "Height", min_value=1, max_value=max(1, image_height - int(edit_y)),
                value=max(1, min(int(pending_bbox[3] - pending_bbox[1]),
                                 image_height - int(edit_y))),
                step=1, key=f"manual_h_{image_hash}_{box_key}"
            )

        adjusted_bbox = [
            int(edit_x), int(edit_y),
            min(image_width, int(edit_x) + int(edit_w)),
            min(image_height, int(edit_y) + int(edit_h))
        ]
        st.session_state.pending_manual_bbox = adjusted_bbox

        # Preview the adjusted rectangle.
        box_preview = image.copy().convert("RGB")
        preview_draw = ImageDraw.Draw(box_preview)
        preview_draw.rectangle(
            adjusted_bbox, outline="#00AA66",
            width=max(3, image_width // 350)
        )
        st.image(
            box_preview,
            caption="Current missing-object box preview",
            use_container_width=True
        )

        if st.button(
            "🗑️ Delete Bounding Box",
            key=f"delete_pending_box_{image_hash}",
            use_container_width=True
        ):
            st.session_state.pending_manual_bbox = None
            st.session_state.manual_canvas_revision += 1
            st.rerun()
    else:
        st.info(
            "Draw one rectangle first. You can then move, resize, "
            "or delete it before saving."
        )

    # Inputs are intentionally outside a form so a Streamlit rerun cannot
    # discard the selected bounding box when the Add button is pressed.
    manual_class = st.text_input(
        "Object label / class",
        placeholder="Example: dog, traffic cone, handbag, tree",
        key=f"manual_class_input_{image_hash}"
    )
    manual_description = st.text_area(
        "Object description",
        placeholder="Write a description of the missing object.",
        key=f"manual_description_input_{image_hash}"
    )

    if st.button("💾 Save Missing Object to Final Review", type="primary", use_container_width=True):
        pending_bbox = st.session_state.pending_manual_bbox
        if not pending_bbox:
            st.error("Draw a bounding box on the image first.")
        elif not manual_class.strip():
            st.error("Enter the object's label or class.")
        else:
            class_name = manual_class.strip().lower()
            new_obj = {
                "id": uid(),
                "label": next_label(class_name, st.session_state.objects),
                "class_name": class_name,
                "confidence": None,
                "bbox": list(pending_bbox),
                "description": manual_description.strip(),
                "description_source": "manual" if manual_description.strip() else "",
                "source": "Manual"
            }
            st.session_state.objects.append(new_obj)
            if not new_obj["description"]:
                new_obj["description"] = describe_object(
                    len(st.session_state.objects) - 1,
                    st.session_state.objects,
                    image_width,
                    image_height
                )
                new_obj["description_source"] = "generated"
            st.session_state.pending_manual_bbox = None
            st.session_state.manual_canvas_revision += 1
            set_annotations_changed()
            st.success(f"{new_obj['label']} added to final annotations.")
            st.rerun()


# =========================================================
# TAB 3: REVIEW + DOWNLOAD
# =========================================================
with tab_review:
    st.subheader("Final Object Labels and Descriptions")
    st.caption("YOLO detections and manually added objects are combined here.")

    if not st.session_state.objects:
        st.info("No objects yet. Detect objects or add them manually.")
    else:
        st.markdown(
            f"**Total objects: {len(st.session_state.objects)}**  ·  "
            f"Manual: {sum(x.get('source') == 'Manual' for x in st.session_state.objects)}"
        )

        for obj in list(st.session_state.objects):
            with st.expander(f"{obj['label']} · {obj['source']}", expanded=False):
                st.write(f"**Class:** {obj['class_name']}")
                st.write(
                    f"**Confidence:** {obj['confidence']:.3f}"
                    if obj.get("confidence") is not None
                    else "**Confidence:** Manual"
                )

                with st.form(f"edit_{obj['id']}"):
                    label_value = st.text_input("Label", value=obj["label"])
                    class_value = st.text_input("Class", value=obj["class_name"])
                    description_value = st.text_area(
                        "Description", value=obj.get("description", "")
                    )
                    save_details = st.form_submit_button("Save changes")

                if save_details:
                    if not label_value.strip() or not class_value.strip():
                        st.error("Label and class are required.")
                    else:
                        obj["label"] = label_value.strip()
                        obj["class_name"] = class_value.strip().lower()
                        obj["description"] = description_value.strip()
                        obj["description_source"] = "manual"
                        set_annotations_changed()
                        st.success("Object updated.")
                        st.rerun()

                if st.button("🗑️ Delete object", key=f"delete_{obj['id']}"):
                    st.session_state.last_deleted = obj.copy()
                    st.session_state.objects = [
                        item for item in st.session_state.objects
                        if item["id"] != obj["id"]
                    ]
                    set_annotations_changed()
                    st.rerun()

        if st.session_state.last_deleted:
            st.caption(f"Last deleted: {st.session_state.last_deleted['label']}")
            if st.button("↶ Undo last delete"):
                restored = st.session_state.last_deleted.copy()
                if any(x["id"] == restored["id"] for x in st.session_state.objects):
                    restored["id"] = uid()
                st.session_state.objects.append(restored)
                st.session_state.last_deleted = None
                set_annotations_changed()
                st.rerun()

        st.divider()
        st.subheader("Final Labelled Image")
        final_image = render_labelled_image(image, st.session_state.objects)
        st.image(final_image, use_container_width=True)

        # Generate fresh export data directly from the current object list.
        # This ensures manually created boxes are included in all downloads.
        export_payload = {
            "image_name": uploaded.name,
            "image_width": image_width,
            "image_height": image_height,
            "scene_description": st.session_state.scene_description,
            "objects": st.session_state.objects
        }
        json_bytes = json.dumps(
            export_payload, ensure_ascii=False, indent=2
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download JSON",
            data=json_bytes,
            file_name="annotations.json",
            mime="application/json",
            use_container_width=True
        )

        class_names = sorted({x["class_name"] for x in st.session_state.objects})
        class_ids = {name: i for i, name in enumerate(class_names)}
        yolo_lines = []
        for obj in st.session_state.objects:
            x1, y1, x2, y2 = obj["bbox"]
            cx = ((x1 + x2) / 2) / image_width
            cy = ((y1 + y2) / 2) / image_height
            bw = (x2 - x1) / image_width
            bh = (y2 - y1) / image_height
            yolo_lines.append(
                f"{class_ids[obj['class_name']]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            )

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("labels.txt", "\n".join(yolo_lines))
            archive.writestr("classes.txt", "\n".join(class_names))
            archive.writestr("annotations.json", json_bytes)

        st.download_button(
            "⬇️ Download YOLO ZIP",
            data=zip_buffer.getvalue(),
            file_name="yolo_annotations.zip",
            mime="application/zip",
            use_container_width=True
        )

        png_buffer = io.BytesIO()
        final_image.save(png_buffer, format="PNG")
        st.download_button(
            "⬇️ Download Labelled Image",
            data=png_buffer.getvalue(),
            file_name="labelled_image.png",
            mime="image/png",
            use_container_width=True
        )

st.caption(
    "Review all detections and manually add any missed objects before exporting. "
    "YOLO detects only the classes it was trained to recognize."
)
