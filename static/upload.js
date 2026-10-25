// Upload behaviour for the viewer: drag-and-drop, preview, loading state.
// Plain JavaScript, no framework. The server still validates every upload;
// the checks here only save a round trip.
(function () {
  const form = document.getElementById("upload-form");
  if (!form) return;

  const viewer = document.getElementById("viewer");
  const input = document.getElementById("file-input");
  const preview = document.getElementById("viewer-preview");
  const prompt = document.getElementById("viewer-prompt");
  const status = document.getElementById("viewer-status");
  const fileName = document.getElementById("file-name");
  const message = document.getElementById("form-message");
  const button = document.getElementById("submit-btn");

  const ALLOWED = /\.(jpe?g|png)$/i;
  let previewUrl = null;

  function showMessage(text) {
    message.textContent = text;
    message.hidden = !text;
  }

  function setFile(file) {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      previewUrl = null;
    }
    if (!file) {
      preview.hidden = true;
      prompt.hidden = false;
      fileName.textContent = "No file chosen";
      return;
    }
    if (!ALLOWED.test(file.name)) {
      input.value = "";
      setFile(null);
      showMessage("\u201C" + file.name + "\u201D isn\u2019t a supported type. Choose a .jpg, .jpeg, or .png file.");
      return;
    }
    showMessage("");
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    preview.alt = "Preview of " + file.name;
    preview.hidden = false;
    prompt.hidden = true;
    fileName.textContent = file.name;
  }

  function clearBusy() {
    viewer.classList.remove("is-busy");
    status.textContent = "";
    button.disabled = false;
    button.textContent = "Run both models";
  }

  input.addEventListener("change", () => setFile(input.files[0]));

  ["dragenter", "dragover"].forEach((type) => {
    viewer.addEventListener(type, (e) => {
      e.preventDefault();
      viewer.classList.add("is-dragging");
    });
  });
  ["dragleave", "drop"].forEach((type) => {
    viewer.addEventListener(type, (e) => {
      e.preventDefault();
      viewer.classList.remove("is-dragging");
    });
  });
  viewer.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length) {
      input.files = files;
      setFile(files[0]);
    }
  });

  // A file dropped outside the viewer would make the browser open it and
  // leave the page, so ignore drops everywhere else.
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  form.addEventListener("submit", (e) => {
    if (!input.files.length) {
      e.preventDefault();
      showMessage("Choose a .jpg, .jpeg, or .png file first.");
      return;
    }
    viewer.classList.add("is-busy");
    status.textContent = "Running both models\u2026";
    button.disabled = true;
    button.textContent = "Running\u2026";
  });

  // Coming back with the browser's Back button should not leave the page stuck.
  window.addEventListener("pageshow", (e) => {
    if (e.persisted) clearBusy();
  });

  // After a result renders, bring it into view.
  const results = document.getElementById("results");
  if (results) {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    results.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
  }
})();