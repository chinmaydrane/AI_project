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

    // Show preview of uploaded image
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
                let output = `
                    🩺 Prediction: <b>${data.result}</b><br>
                    🔹 Confidence: ${data.confidence}%<br>
                `;

                // If TB positive, show lesion details and overlay
                if (data.result === "Positive") {
                    output += `
                        💀 Lung Damage: ${data.damage_percent}%<br>
                        🫁 ${data.region_summary}<br>
                    `;

                    if (data.overlay_path) {
                        output += `<img src="${data.overlay_path}" alt="Lesion Overlay" 
                                    style="width:100%;margin-top:10px;border-radius:8px;">`;
                    }
                }

                resultBox.innerHTML = output;
            }
        })
        .catch((err) => {
            resultBox.innerHTML = "⚠️ Error: " + err.message;
        });
}
