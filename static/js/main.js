/* ------------------------------------------------------------------
   DeepGuard - Deepfake Detection UI Logic
   Handles file upload, progress simulation, and results rendering.
   ------------------------------------------------------------------ */

(function () {
    "use strict";

    // DOM elements
    const dropZone = document.getElementById("drop-zone");
    const videoInput = document.getElementById("video-input");
    const uploadForm = document.getElementById("upload-form");
    const fileInfo = document.getElementById("file-info");
    const fileNameEl = document.getElementById("file-name");
    const removeFileBtn = document.getElementById("remove-file");
    const analyzeBtn = document.getElementById("analyze-btn");
    const btnText = document.querySelector(".btn-text");
    const progressArea = document.getElementById("progress-area");
    const progressFill = document.getElementById("progress-fill");
    const progressText = document.getElementById("progress-text");
    const results = document.getElementById("results");
    const uploadIcon = document.getElementById("upload-icon");
    const dropTitle = document.getElementById("drop-title");
    const fileHint = document.getElementById("file-hint");
    const tabVideo = document.getElementById("tab-video");
    const tabImage = document.getElementById("tab-image");

    // Result elements
    const resultIcon = document.getElementById("result-icon");
    const resultLabel = document.getElementById("result-label");
    const resultConfidence = document.getElementById("result-confidence");
    const ringFg = document.getElementById("ring-fg");
    const ringValue = document.getElementById("ring-value");
    const realBar = document.getElementById("real-bar");
    const fakeBar = document.getElementById("fake-bar");
    const realPct = document.getElementById("real-pct");
    const fakePct = document.getElementById("fake-pct");

    let selectedFile = null;
    let progressInterval = null;
    let mediaMode = "video"; // "video" or "image"

    const VIDEO_EXTENSIONS = ["mp4", "avi", "mov", "mkv", "webm", "m4v"];
    const IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "bmp", "webp", "tiff"];

    // ---------- Media mode toggle ----------

    function setMediaMode(mode) {
        mediaMode = mode;
        const isVideo = mode === "video";

        tabVideo.classList.toggle("active", isVideo);
        tabImage.classList.toggle("active", !isVideo);

        // Update labels
        uploadIcon.textContent = isVideo ? "📹" : "🖼️";
        dropTitle.textContent = isVideo
            ? "Drag & Drop your video here"
            : "Drag & Drop your image here";
        fileHint.textContent = isVideo
            ? "MP4, AVI, MOV, MKV, WEBM · Max 200 MB"
            : "JPG, PNG, BMP, WEBP, TIFF";
        btnText.innerHTML = isVideo
            ? "<svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" width=\"18\" height=\"18\"><circle cx=\"11\" cy=\"11\" r=\"7\"/><path d=\"M21 21l-4.35-4.35M11 8v6M8 11h6\" stroke-linecap=\"round\"/></svg> Analyze Video"
            : "<svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" width=\"18\" height=\"18\"><circle cx=\"11\" cy=\"11\" r=\"7\"/><path d=\"M21 21l-4.35-4.35M11 8v6M8 11h6\" stroke-linecap=\"round\"/></svg> Analyze Image";

        // Update the hidden input accept attribute
        videoInput.accept = isVideo ? "video/*" : "image/*";

        // Reset selected file
        removeFileBtn.click();
    }

    tabVideo.addEventListener("click", () => setMediaMode("video"));
    tabImage.addEventListener("click", () => setMediaMode("image"));

    // ---------- File selection handlers ----------

    dropZone.addEventListener("click", () => videoInput.click());
    dropZone.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            videoInput.click();
        }
    });
    dropZone.setAttribute("tabindex", "0");
    dropZone.setAttribute("role", "button");

    videoInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) setSelectedFile(file);
    });

    // Drag and drop
    ["dragenter", "dragover"].forEach((evt) => {
        dropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            dropZone.classList.add("dragover");
        });
    });
    ["dragleave", "drop"].forEach((evt) => {
        dropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            dropZone.classList.remove("dragover");
        });
    });
    dropZone.addEventListener("drop", (e) => {
        const file = e.dataTransfer.files[0];
        if (file) setSelectedFile(file);
    });

    function setSelectedFile(file) {
        const isVideo = mediaMode === "video";
        const allowed = isVideo ? VIDEO_EXTENSIONS : IMAGE_EXTENSIONS;
        const ext = file.name.split(".").pop().toLowerCase();
        if (!allowed.includes(ext)) {
            alert(
                isVideo
                    ? "Unsupported file type. Please use: MP4, AVI, MOV, MKV, WEBM, M4V"
                    : "Unsupported file type. Please use: JPG, PNG, BMP, WEBP, TIFF"
            );
            return;
        }
        if (file.size > 200 * 1024 * 1024) {
            alert("File too large. Maximum allowed size is 200 MB.");
            return;
        }
        selectedFile = file;
        fileNameEl.textContent = file.name;
        fileInfo.style.display = "flex";
        analyzeBtn.disabled = false;
    }

    removeFileBtn.addEventListener("click", () => {
        selectedFile = null;
        videoInput.value = "";
        fileInfo.style.display = "none";
        analyzeBtn.disabled = true;
    });

    // ---------- Form submission ----------

    uploadForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (!selectedFile) return;

        // Reset results
        results.style.display = "none";
        resetProgress();

        const isVideo = mediaMode === "video";
        const formData = new FormData();
        formData.append(isVideo ? "video" : "image", selectedFile);

        analyzeBtn.disabled = true;
        btnText.textContent = "Analyzing...";
        progressArea.style.display = "block";
        progressFill.style.width = "5%";
        progressText.textContent = isVideo ? "Uploading video..." : "Uploading image...";

        // Simulate indeterminate progress while request runs
        startProgressSimulation();

        try {
            const endpoint = isVideo ? "/api/predict" : "/api/predict_image";
            const response = await fetch(endpoint, {
                method: "POST",
                body: formData,
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Analysis failed. Please try again.");
            }

            stopProgressSimulation();
            progressFill.style.width = "100%";
            progressText.textContent = "Analysis complete!";
            renderResults(data);
        } catch (err) {
            stopProgressSimulation();
            progressArea.style.display = "none";
            alert("Error: " + err.message);
        } finally {
            analyzeBtn.disabled = false;
            const label = isVideo ? "Video" : "Image";
            btnText.innerHTML = "<svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" width=\"18\" height=\"18\"><circle cx=\"11\" cy=\"11\" r=\"7\"/><path d=\"M21 21l-4.35-4.35M11 8v6M8 11h6\" stroke-linecap=\"round\"/></svg> Analyze " + label;
        }
    });

    // ---------- Progress simulation ----------

    function startProgressSimulation() {
        let progress = 5;
        stopProgressSimulation();
        progressInterval = setInterval(() => {
            progress += Math.random() * 8;
            if (progress > 90) progress = 90;
            progressFill.style.width = progress + "%";
            if (progress < 35) {
                progressText.textContent = "Extracting frames...";
            } else if (progress < 65) {
                progressText.textContent = "Running CNN feature extraction...";
            } else if (progress < 88) {
                progressText.textContent = "Analyzing temporal patterns with RNN...";
            }
        }, 600);
    }

    function stopProgressSimulation() {
        if (progressInterval) {
            clearInterval(progressInterval);
            progressInterval = null;
        }
    }

    function resetProgress() {
        progressFill.style.width = "0%";
        progressText.textContent = "";
    }

    // ---------- Results rendering ----------

    function renderResults(data) {
        const isFake = data.prediction === "FAKE";

        results.style.display = "block";

        resultIcon.textContent = isFake ? "🚨" : "✅";
        resultLabel.textContent = data.prediction;
        resultConfidence.textContent = "Confidence: " + data.confidence + "%";

        resultLabel.style.color = isFake ? "var(--red)" : "var(--green)";
        ringFg.style.stroke = isFake ? "var(--red)" : "var(--green)";

        // Animate ring
        const offset = 327 - (data.confidence / 100) * 327;
        ringFg.style.strokeDashoffset = offset;
        ringValue.textContent = data.confidence + "%";

        // Bars
        realBar.style.width = data.real_probability + "%";
        fakeBar.style.width = data.fake_probability + "%";
        realPct.textContent = data.real_probability + "%";
        fakePct.textContent = data.fake_probability + "%";

        results.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
})();
