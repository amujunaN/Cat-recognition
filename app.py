
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time


# --------------------------------
# 1. Page configuration
# --------------------------------

st.set_page_config(
    page_title="Cat Recognition",
    page_icon="🐱",
    layout="wide"
)

st.title("🐱 Cat Recognition System")
st.write(
    "Upload an image to classify it as Cat or Non-cat "
    "and visualize the model's attention using Grad-CAM."
)


# --------------------------------
# 2. Load the trained model
# --------------------------------

@st.cache_resource
def load_trained_model():
    return keras.models.load_model(
        "efficientnetv2_cats.keras"
    )


model = load_trained_model()


# --------------------------------
# 3. Find EfficientNet backbone
# --------------------------------

# The backbone is nested inside the trained model
base_model = next(
    layer for layer in model.layers
    if isinstance(layer, keras.Model)
)

# Find the last convolutional layer
def get_all_layers(model):
    for layer in model.layers:
        yield layer

        if isinstance(layer, keras.Model):
            yield from get_all_layers(layer)


all_layers = list(get_all_layers(base_model))

# Сначала ищем Conv2D и DepthwiseConv2D
conv_layers = [
    layer for layer in all_layers
    if isinstance(
        layer,
        (keras.layers.Conv2D, keras.layers.DepthwiseConv2D)
    )
]

if conv_layers:
    last_conv_layer = conv_layers[-1]
else:
    # Если свёрточные слои не найдены,
    # ищем последний слой с пространственной картой признаков
    spatial_layers = []

    for layer in all_layers:
        try:
            if len(layer.output.shape) == 4:
                spatial_layers.append(layer)
        except (AttributeError, ValueError):
            pass

    if not spatial_layers:
        raise ValueError("Could not find a suitable Grad-CAM layer.")

    last_conv_layer = spatial_layers[-1]

print("Grad-CAM layer:", last_conv_layer.name)

# Use the final spatial feature layer of EfficientNetV2B0
last_conv_layer = base_model.get_layer("top_activation")

print("Grad-CAM layer:", last_conv_layer.name)
print("Layer output shape:", last_conv_layer.output.shape)

# Build Grad-CAM model
grad_model = keras.Model(
    inputs=base_model.inputs,
    outputs=[
        last_conv_layer.output,
        base_model.output
    ]
)


# --------------------------------
# 4. Prepare Grad-CAM model
# --------------------------------

grad_model = keras.Model(
    inputs=base_model.input,
    outputs=[
        last_conv_layer.output,
        base_model.output
    ]
)

pooling_layer = model.get_layer(
    "global_average_pooling"
)

dropout_layer = model.get_layer(
    "efficient_dropout"
)

prediction_layer = model.get_layer(
    "efficient_prediction"
)


# --------------------------------
# 5. Grad-CAM function
# --------------------------------

def make_gradcam_heatmap(image, class_index):

    image = tf.convert_to_tensor(
        image,
        dtype=tf.float32
    )

    image = tf.expand_dims(image, axis=0)

    with tf.GradientTape() as tape:

        conv_outputs, backbone_output = grad_model(
            image,
            training=False
        )

        x = pooling_layer(backbone_output)
        x = dropout_layer(x, training=False)

        predictions = prediction_layer(x)

        if class_index == 1:
            score = predictions[:, 0]
        else:
            score = 1.0 - predictions[:, 0]

    gradients = tape.gradient(
        score,
        conv_outputs
    )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    heatmap = heatmap / (
        tf.reduce_max(heatmap) + keras.backend.epsilon()
    )

    return heatmap.numpy()


# --------------------------------
# 6. Image upload
# --------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    original_image = Image.open(
        uploaded_file
    ).convert("RGB")

    resized_image = original_image.resize(
        (224, 224)
    )

    image_array = np.array(
        resized_image,
        dtype=np.float32
    )

    # --------------------------------
    # 7. Prediction and latency
    # --------------------------------

    if st.button("Predict", type="primary"):

        start_time = time.perf_counter()

        probability = model.predict(
            np.expand_dims(image_array, axis=0),
            verbose=0
        )[0][0]

        predicted_class = int(probability >= 0.5)

        confidence = (
            1 - probability
            if predicted_class == 0
            else probability
        )

        class_names = ["Cat", "Non-cat"]

        predicted_name = class_names[predicted_class]

        # Generate Grad-CAM
        heatmap = make_gradcam_heatmap(
            image_array,
            predicted_class
        )

        end_time = time.perf_counter()

        latency = (end_time - start_time) * 1000

        # --------------------------------
        # 8. Display prediction
        # --------------------------------

        st.subheader("Prediction result")

        if predicted_class == 0:
            st.success("Prediction: Cat 🐱")
        else:
            st.warning("Prediction: Non-cat")

        st.metric(
            "Model confidence",
            f"{confidence:.2%}"
        )

        st.metric(
            "Inference time",
            f"{latency:.2f} ms"
        )

        # --------------------------------
        # 9. Prepare Grad-CAM overlay
        # --------------------------------

        heatmap_resized = tf.image.resize(
            heatmap[..., np.newaxis],
            (224, 224)
        ).numpy().squeeze()

        heatmap_colored = plt.get_cmap("jet")(
            heatmap_resized
        )[:, :, :3]

        original_array = image_array / 255.0

        overlay = (
            0.6 * original_array
            + 0.4 * heatmap_colored
        )

        overlay = np.clip(
            overlay, 0, 1
        )

        # --------------------------------
        # 10. Display images
        # --------------------------------

        st.subheader("Visual explanation: Grad-CAM")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                original_image,
                caption="Original image",
                use_container_width=True
            )

        with col2:
            st.image(
                overlay,
                caption="Grad-CAM heatmap overlay",
                use_container_width=True
            )

        st.caption(
            "The heatmap highlights image regions "
            "that contributed to the selected prediction. "
            "It is an explanation aid, not proof of model reasoning."
        )

else:
    st.info("Please upload an image to begin.")
