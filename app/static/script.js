const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const removeFile = document.getElementById('removeFile');
const submitBtn = document.getElementById('submitBtn');
const form = document.querySelector('.upload-form');

// Drag and drop functionality
uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files;
        handleFileSelect();
    }
});

fileInput.addEventListener('change', handleFileSelect);
removeFile.addEventListener('click', clearFile);

function handleFileSelect() {
    const file = fileInput.files[0];
    if (file) {
        fileName.textContent = file.name;
        uploadArea.style.display = 'none';
        fileInfo.style.display = 'flex';
        submitBtn.disabled = false;
        submitBtn.classList.add('enabled');
    }
}

function clearFile() {
    fileInput.value = '';
    uploadArea.style.display = 'flex';
    fileInfo.style.display = 'none';
    submitBtn.disabled = true;
    submitBtn.classList.remove('enabled');
}

// Form submission with loading state
form.addEventListener('submit', () => {
    const btnText = document.querySelector('.btn-text');
    const btnLoading = document.querySelector('.btn-loading');
    btnText.style.display = 'none';
    btnLoading.style.display = 'inline-flex';
    submitBtn.disabled = true;
});

// Add entrance animations
document.addEventListener('DOMContentLoaded', () => {
    document.body.classList.add('loaded');
});
