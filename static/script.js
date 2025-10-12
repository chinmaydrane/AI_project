function uploadImage() {
    const fileInput = document.getElementById("imageUpload");
    const file = fileInput.files[0];
    const resultBox = document.getElementById("result-box");
    const preview = document.getElementById("preview");

    if (!file) {
        alert("Please select an image first.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    // Preview image
    const reader = new FileReader();
    reader.onload = function (e) {
        preview.innerHTML = `<img src="${e.target.result}" alt="Uploaded Image"/>`;
    };
    reader.readAsDataURL(file);

    resultBox.innerHTML = "⏳ Processing...";

    fetch("/predict", {
        method: "POST",
        body: formData,
    })
        .then((response) => response.json())
        .then((data) => {
            if (data.error) {
                resultBox.innerHTML = "❌ " + data.error;
            } else {
                resultBox.innerHTML = `🩺 Prediction: <b>${data.result}</b><br>🔹 Confidence: ${data.confidence}%`;
            }
        })
        .catch((err) => {
            resultBox.innerHTML = "⚠️ Error: " + err.message;
        });
}
