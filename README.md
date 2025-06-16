# Health Disease Prediction System

A professional and comprehensive health disease prediction system that provides accurate disease predictions, nearby hospital locations, and treatment recommendations based on user health parameters.

## Features

- Advanced disease prediction using machine learning algorithms
- Real-time location detection and geolocation services
- Nearby hospital suggestions with detailed information
- Comprehensive disease information database
- Evidence-based treatment recommendations
- Professional and user-friendly interface
- Secure API endpoints with rate limiting
- Comprehensive logging and error handling
- Configurable application settings

## Prerequisites

- Python 3.7 or higher
- Google Places API key (for hospital location features)
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd health-disease-prediction
```

2. Create and activate a virtual environment:
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On Unix or MacOS
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure the application:
   - Copy `config.ini.example` to `config.ini`
   - Update the configuration values in `config.ini`
   - Create a `.env` file with your API keys:
     ```
     GOOGLE_PLACES_API_KEY=your_api_key_here
     SECRET_KEY=your_secret_key_here
     ```

## Configuration

The application can be configured through `config.ini`:

- `[app]`: Basic application settings
- `[api]`: API configuration and rate limiting
- `[logging]`: Logging configuration
- `[security]`: Security settings
- `[features]`: Feature toggles
- `[limits]`: Application limits

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Access the application at `http://localhost:5000`

3. Use the health checker form to:
   - Enter your health parameters
   - Select symptoms
   - Add existing conditions
   - Get personalized predictions

## API Endpoints

### Health Prediction
- `POST /predict`: Submit health data for disease prediction
- Rate limit: 30 requests per minute

### Location Services
- `POST /search_locations`: Search for locations
- `POST /get_hospitals`: Get nearby hospitals
- Rate limit: 50 requests per minute

### Chat Interface
- `POST /chat`: Interact with the health assistant
- Rate limit: 50 requests per minute

## Security Considerations

- All API endpoints are rate-limited
- Input validation and sanitization
- Secure configuration management
- Error handling and logging
- CORS protection
- Request size limits

## Development

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all functions
- Maintain comprehensive logging

### Testing
```bash
# Run tests
python -m pytest

# Run with coverage
python -m pytest --cov=.
```

## Deployment

1. Set up a production environment:
   - Use a production-grade WSGI server (e.g., Gunicorn)
   - Configure a reverse proxy (e.g., Nginx)
   - Set up SSL/TLS certificates

2. Update configuration:
   - Set `debug = False` in `config.ini`
   - Configure proper logging
   - Set secure secret keys

3. Deploy using your preferred method:
   - Docker
   - Traditional deployment
   - Cloud platform

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

## Disclaimer

This application is for educational and informational purposes only. Always consult with healthcare professionals for medical advice and treatment. The predictions and recommendations provided by this system should not be considered as medical advice. 