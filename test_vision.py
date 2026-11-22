
import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO
import ollama
import json
import gc
import io

# ---------------- CONFIG ----------------
AI_MODEL = "qwen3-vl:2b"
YOLO_MODEL = "yolo11n.pt"

st.set_page_config(
    page_title="AI Image Annotation",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ AI Image Annotation Platform")
st.caption("Object Detection | AI Descriptions | Manual Annotation | Export")

# ---------------- SESSION STATE ----------------
defaults = {
    "objects": [],
    "scene_description": "",
    "annotated_image": None,
    "image_bytes": None,
    "analysis_done": False
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------- IMAGE FUNCTIONS ----------------
def resize_image(image, max_size=448):
    img = image.copy()
    img.thumbnail((max_size, max_size))
    return img


def draw_boxes(image, objects):
    result = image.copy().convert("RGB")
    draw = ImageDraw.Draw(result)

    for obj in objects:
        x1, y1, x2, y2 = obj["box"]

        draw.rectangle(
            (x1, y1, x2, y2),
            outline="red",
            width=3
        )

        draw.text(
            (x1, max(0, y1 - 18)),
            obj["id"],
            fill="red"
        )

    return result


# ---------------- YOLO DETECTION ----------------
def detect_objects(image):
    model = YOLO(YOLO_MODEL)

    results = model.predict(
        source=image,
        imgsz=416,
        conf=0.25,
        max_det=25,
        device="cpu",
        verbose=False
    )

    detected = []
    counters = {}

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            counters[class_name] = counters.get(class_name, 0) + 1

            object_id = f"{class_name} {counters[class_name]}"

            coords = box.xyxy[0].tolist()

            detected.append({
                "id": object_id,
                "label": class_name,
                "box": [int(v) for v in coords],
                "confidence": round(float(box.conf[0]), 2),
                "description": ""
            })

    del model
    del results
    gc.collect()

    return detected


# ---------------- AI DESCRIPTIONS ----------------
def generate_descriptions(image, objects):
    if not objects:
        return [], "No objects were detected in this image."

    small_image = resize_image(image, 448)

    image_buffer = io.BytesIO()
    small_image.save(
        image_buffer,
        format="JPEG",
        quality=65
    )

    image_data = image_buffer.getvalue()

    object_list = []

    for obj in objects:
        object_list.append(
            f'{obj["id"]}: label={obj["label"]}, '
            f'bounding_box={obj["box"]}'
        )

    prompt = f"""
Analyze the supplied image carefully.

Detected objects from the object detection system:
{chr(10).join(object_list)}

Return valid JSON only, using this structure:

{{
  "objects": [
    {{
      "id": "Person 1",
      "description": "Describe only visible details of this object."
    }}
  ],
  "scene_description": "Describe the complete scene."
}}

Rules:
1. Include every supplied object ID exactly once.
2. Describe each object separately.
3. For people, describe visible clothing, posture and action only.
4. For other objects, describe their visible appearance and position.
5. Mention relationships between objects only when clearly visible.
6. Do not guess hidden details or identity.
7. If an object is unclear, say that its details are unclear.
8. The scene description should summarize the whole image.
9. Do not add objects that are not visible.
"""

    response = ollama.chat(
        model=AI_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [image_data]
            }
        ],
        format="json",
        options={
            "temperature": 0.1,
            "num_predict": 180,
            "num_ctx": 1024,
            "num_thread": 2
        },
        keep_alive=0
    )

    answer = response["message"]["content"]
    data = json.loads(answer)

    descriptions = {
        item.get("id"): item.get("description", "")
        for item in data.get("objects", [])
    }

    for obj in objects:
        obj["description"] = descriptions.get(
            obj["id"],
            "Description not available."
        )

    scene = data.get(
        "scene_description",
        "Scene description not available."
    )

    return objects, scene


# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    if st.session_state.image_bytes != uploaded_file.getvalue():
        st.session_state.objects = []
        st.session_state.scene_description = ""
        st.session_state.annotated_image = None
        st.session_state.analysis_done = False
        st.session_state.image_bytes = uploaded_file.getvalue()

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    # ---------------- DETECTION BUTTON ----------------
    if st.button("🔍 Detect Objects", use_container_width=True):
        with st.spinner("Detecting objects..."):
            try:
                st.session_state.objects = detect_objects(image)
                st.session_state.annotated_image = draw_boxes(
                    image,
                    st.session_state.objects
                )
                st.session_state.analysis_done = False
                st.success(
                    f'{len(st.session_state.objects)} objects detected.'
                )
            except Exception as e:
                st.error(f"Detection error: {e}")

    # ---------------- AI DESCRIPTION BUTTON ----------------
    if st.session_state.objects:
        if st.button(
            "✨ Generate Object & Scene Descriptions",
            use_container_width=True
        ):
            with st.spinner(
                "AI is analyzing the image. This may take some time..."
            ):
                try:
                    updated_objects, scene = generate_descriptions(
                        image,
                        st.session_state.objects
                    )

                    st.session_state.objects = updated_objects
                    st.session_state.scene_description = scene
                    st.session_state.analysis_done = True

                    st.success("AI descriptions generated!")

                except Exception as e:
                    st.error(f"AI description error: {e}")

    # ---------------- DISPLAY DETECTIONS ----------------
    if st.session_state.objects:
        st.divider()
        st.subheader("📦 Detected Objects")

        for i, obj in enumerate(st.session_state.objects):
            with st.expander(
                f'{obj["id"]} | Confidence: {obj["confidence"]}'
            ):
                st.write("**Object Label:**", obj["label"])
                st.write("**Bounding Box:**", obj["box"])
                st.write("**AI Description:**")

                if obj["description"]:
                    st.write(obj["description"])
                else:
                    st.info("Generate AI descriptions to view.")

        # ---------------- SCENE DESCRIPTION ----------------
        if st.session_state.scene_description:
            st.divider()
            st.subheader("🌍 Complete Scene Description")
            st.write(st.session_state.scene_description)

        # ---------------- MANUAL ANNOTATION ----------------
        st.divider()
        st.subheader("✏️ Manual Annotation")

        with st.form("manual_annotation"):
            new_label = st.text_input(
                "Object name",
                placeholder="e.g. Chair"
            )

            col1, col2 = st.columns(2)

            with col1:
                x1 = st.number_input("X1", min_value=0, value=0)
                y1 = st.number_input("Y1", min_value=0, value=0)

            with col2:
                x2 = st.number_input(
                    "X2", min_value=1, value=max(1, image.width // 4)
                )
                y2 = st.number_input(
                    "Y2", min_value=1, value=max(1, image.height // 4)
                )

            manual_description = st.text_area(
                "Object description"
            )

            add_object = st.form_submit_button("Add Object")

            if add_object:
                if not new_label.strip():
                    st.error("Please enter an object name.")
                elif x2 <= x1 or y2 <= y1:
                    st.error("Please enter valid box coordinates.")
                elif x2 > image.width or y2 > image.height:
                    st.error("Coordinates exceed image dimensions.")
                else:
                    count = sum(
                        1 for obj in st.session_state.objects
                        if obj["label"].lower() == new_label.strip().lower()
                    ) + 1

                    st.session_state.objects.append({
                        "id": f"{new_label.strip()} {count}",
                        "label": new_label.strip(),
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": 1.0,
                        "description": manual_description
                    })

                    st.session_state.annotated_image = draw_boxes(
                        image,
                        st.session_state.objects
                    )
                    st.rerun()

        # ---------------- EDIT / DELETE ----------------
        st.divider()
        st.subheader("🛠️ Edit or Delete Objects")

        object_ids = [
            obj["id"] for obj in st.session_state.objects
        ]

        selected_id = st.selectbox(
            "Select object",
            object_ids
        )

        selected_obj = next(
            obj for obj in st.session_state.objects
            if obj["id"] == selected_id
        )

        with st.form("edit_object"):
            edited_label = st.text_input(
                "Edit label",
                value=selected_obj["label"]
            )

            edited_description = st.text_area(
                "Edit description",
                value=selected_obj["description"]
            )

            save_edit = st.form_submit_button("Save Changes")

            if save_edit:
                selected_obj["label"] = edited_label
                selected_obj["description"] = edited_description
                st.rerun()

        if st.button("🗑️ Delete Selected Object"):
            st.session_state.objects = [
                obj for obj in st.session_state.objects
                if obj["id"] != selected_id
            ]
            st.rerun()

        # ---------------- FINAL IMAGE ----------------
        final_image = draw_boxes(
            image,
            st.session_state.objects
        )

        st.divider()
        st.subheader("🖼️ Annotated Image")
        st.image(final_image, use_container_width=True)

        # ---------------- JSON EXPORT ----------------
        export_data = {
            "objects": st.session_state.objects,
            "scene_description": st.session_state.scene_description
        }

        json_data = json.dumps(
            export_data,
            indent=4,
            ensure_ascii=False
        )

        st.download_button(
            "⬇️ Download JSON",
            data=json_data,
            file_name="annotations.json",
            mime="application/json"
        )

        # ---------------- YOLO EXPORT ----------------
        class_names = sorted(set(
            obj["label"] for obj in st.session_state.objects
        ))

        class_map = {
            name: index for index, name in enumerate(class_names)
        }

        yolo_lines = []

        for obj in st.session_state.objects:
            x1, y1, x2, y2 = obj["box"]

            center_x = ((x1 + x2) / 2) / image.width
            center_y = ((y1 + y2) / 2) / image.height
            width = (x2 - x1) / image.width
            height = (y2 - y1) / image.height

            yolo_lines.append(
                f'{class_map[obj["label"]]} '
                f'{center_x:.6f} {center_y:.6f} '
                f'{width:.6f} {height:.6f}'
            )

        st.download_button(
            "⬇️ Download YOLO TXT",
            data="\n".join(yolo_lines),
            file_name="annotations.txt",
            mime="text/plain"
        )

        # ---------------- IMAGE EXPORT ----------------
        output_buffer = io.BytesIO()
        final_image.save(output_buffer, format="JPEG")

        st.download_button(
            "⬇️ Download Annotated Image",
            data=output_buffer.getvalue(),
            file_name="annotated_image.jpg",
            mime="image/jpeg"
        )

else:
    st.info("Upload an image to get started.")
