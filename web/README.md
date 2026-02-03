# WeldNet Web Interface

This directory contains the web interface for WeldNet, providing an interactive demonstration and documentation viewer.

## Contents

- `templates/` - HTML pages
  - `index.html` - Home page
  - `demo.html` - Interactive demo interface
  - `documentation.html` - Documentation page
  - `about.html` - About page

- `static/` - Static assets
  - `css/style.css` - Stylesheet
  - `js/demo.js` - Demo interface JavaScript

## Running Locally

To view the web interface locally:

### Option 1: Simple HTTP Server (Python)

```bash
cd web/templates
python -m http.server 8000
```

Then open http://localhost:8000 in your browser.

### Option 2: Using any static file server

Any static file server can be used to serve the HTML files. For example:

```bash
# Using npm's http-server
npx http-server web/templates -p 8000

# Using PHP
cd web/templates
php -S localhost:8000
```

## Features

### Home Page
- Overview of WeldNet features
- Model variants comparison
- Quick start guide
- Links to documentation and GitHub

### Demo Page
- Interactive image upload interface
- Model variant selection
- Mock defect detection (simulation)
- Results visualization with probabilities

**Note:** The demo page simulates predictions for demonstration purposes. To use actual inference, deploy a backend service using the Python API.

### Documentation Page
- Installation instructions
- Dataset preparation guide
- Training and inference examples
- API reference

### About Page
- Project overview
- Architecture highlights
- Technical specifications
- Use cases and applications
- Citation information

## Integration with Backend

To integrate the demo with a real backend:

1. Create a Flask/FastAPI backend service:

```python
from flask import Flask, request, jsonify
from inference import load_model, predict_image

app = Flask(__name__)
model = load_model('path/to/model.pth', 'weldnet', 6, 3, 'cpu')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    result = predict_image(model, file, transform, device, class_names)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

2. Update `demo.js` to call the backend API instead of using mock results.

## Deployment

To deploy the web interface:

1. **GitHub Pages**: Push the `web/templates` directory to GitHub Pages
2. **Netlify/Vercel**: Connect your repository and set build directory to `web/templates`
3. **Custom Server**: Serve the HTML files using nginx, Apache, or any web server

## Customization

### Styling
Edit `static/css/style.css` to customize colors, fonts, and layout.

### Content
Edit the HTML files in `templates/` to update content, add sections, or modify text.

### Functionality
Edit `static/js/demo.js` to modify the demo behavior or integrate with a backend.

## Browser Support

The web interface supports all modern browsers:
- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Opera 76+

## License

Same as the main WeldNet project - MIT License
