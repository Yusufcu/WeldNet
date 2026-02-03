// WeldNet Demo Interface JavaScript

// Demo state
let selectedFile = null;
let selectedModel = 'standard';

// DOM elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const modelSelect = document.getElementById('modelSelect');
const analyzeBtn = document.getElementById('analyzeBtn');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const loadingIndicator = document.getElementById('loadingIndicator');
const results = document.getElementById('results');
const tryAnotherBtn = document.getElementById('tryAnotherBtn');

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
});

function setupEventListeners() {
    // Upload area click
    uploadArea.addEventListener('click', function() {
        fileInput.click();
    });

    // Drag and drop
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.style.borderColor = '#2563eb';
        uploadArea.style.backgroundColor = '#f0f9ff';
    });

    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadArea.style.borderColor = '#e2e8f0';
        uploadArea.style.backgroundColor = '';
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.style.borderColor = '#e2e8f0';
        uploadArea.style.backgroundColor = '';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    // File input change
    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Model selection
    modelSelect.addEventListener('change', function(e) {
        selectedModel = e.target.value;
    });

    // Analyze button
    analyzeBtn.addEventListener('click', function() {
        if (selectedFile) {
            analyzeImage();
        }
    });

    // Try another button
    tryAnotherBtn.addEventListener('click', function() {
        resetDemo();
    });
}

function handleFileSelect(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file');
        return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        alert('File size must be less than 10MB');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImg.src = e.target.result;
        imagePreview.style.display = 'block';
        analyzeBtn.disabled = false;
        results.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

function analyzeImage() {
    // Hide previous results
    results.style.display = 'none';
    analyzeBtn.disabled = true;
    
    // Show loading indicator
    loadingIndicator.style.display = 'block';

    // Simulate API call (in production, this would call the actual backend)
    setTimeout(function() {
        // Mock prediction results
        const mockResults = generateMockResults();
        
        // Hide loading
        loadingIndicator.style.display = 'none';
        
        // Show results
        displayResults(mockResults);
        
        // Re-enable button
        analyzeBtn.disabled = false;
    }, 2000);
}

function generateMockResults() {
    // Mock defect classes
    const classes = ['good', 'crack', 'porosity', 'slag', 'lack_of_fusion', 'undercut'];
    
    // Generate random prediction
    const predictedIndex = Math.floor(Math.random() * classes.length);
    const confidence = 0.75 + Math.random() * 0.2; // 75-95%
    
    // Generate probabilities
    const probabilities = classes.map((className, index) => {
        if (index === predictedIndex) {
            return confidence;
        } else {
            return (1 - confidence) / (classes.length - 1);
        }
    });

    return {
        predictedClass: classes[predictedIndex],
        confidence: confidence,
        probabilities: probabilities.map((prob, index) => ({
            className: classes[index],
            probability: prob
        }))
    };
}

function displayResults(results) {
    // Set result class and confidence
    const resultClass = document.getElementById('resultClass');
    const resultConfidence = document.getElementById('resultConfidence');
    const resultIcon = document.getElementById('resultIcon');

    // Format class name
    const formattedClass = results.predictedClass
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');

    resultClass.textContent = formattedClass;
    resultConfidence.textContent = `Confidence: ${(results.confidence * 100).toFixed(1)}%`;

    // Set appropriate icon
    if (results.predictedClass === 'good') {
        resultIcon.textContent = '✅';
    } else {
        resultIcon.textContent = '⚠️';
    }

    // Display all probabilities
    const probabilitiesList = document.getElementById('probabilitiesList');
    probabilitiesList.innerHTML = '';

    // Sort probabilities in descending order
    const sortedProbs = [...results.probabilities].sort((a, b) => b.probability - a.probability);

    sortedProbs.forEach(item => {
        const probItem = document.createElement('div');
        probItem.className = 'probability-item';

        const label = document.createElement('span');
        label.style.minWidth = '150px';
        label.textContent = item.className
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');

        const barContainer = document.createElement('div');
        barContainer.className = 'probability-bar';

        const barFill = document.createElement('div');
        barFill.className = 'probability-fill';
        barFill.style.width = `${item.probability * 100}%`;

        barContainer.appendChild(barFill);

        const percentage = document.createElement('span');
        percentage.style.minWidth = '50px';
        percentage.style.textAlign = 'right';
        percentage.textContent = `${(item.probability * 100).toFixed(1)}%`;

        probItem.appendChild(label);
        probItem.appendChild(barContainer);
        probItem.appendChild(percentage);

        probabilitiesList.appendChild(probItem);
    });

    // Show results
    results.style.display = 'block';
    
    // Scroll to results
    results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function resetDemo() {
    selectedFile = null;
    fileInput.value = '';
    imagePreview.style.display = 'none';
    results.style.display = 'none';
    analyzeBtn.disabled = true;
    previewImg.src = '';
}

// Note display function
function showNote() {
    console.log('WeldNet Demo - This is a simulation interface.');
    console.log('For actual inference, use the Python API or deploy a backend service.');
    console.log('Example: python inference.py --model_path model.pth --input image.jpg');
}

// Call note on load
showNote();
