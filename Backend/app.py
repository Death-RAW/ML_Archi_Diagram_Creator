import os
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from logging.config import dictConfig
from scripts.processor import Processor
from scripts.diagram_generator import DiagramGenerator

# Enable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "true"

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'console': {
        'class': 'logging.StreamHandler',
        'formatter': 'default'
    }},
    'loggers': {
        'gunicorn.error': {
            'level': 'DEBUG',
            'handlers': ['console'],
            'propagate': 0,
            'qualname': 'gunicorn.error'
        },
        'gunicorn.access': {
            'level': 'DEBUG',
            'handlers': ['console'],
            'propagate': 0,
            'qualname': 'gunicorn.access'
        },
        'flask.app': {
            'level': 'DEBUG',
            'handlers': ['console'],
            'propagate': 0,
        },
    },
})


app = Flask(__name__)
CORS(app)

@app.route('/')
def landing():
    return 'Welcome to App'

# Generate the diagram based on the description
@app.route('/api/description', methods=['POST'])
def system_description():
    data = request.get_json()
    app.logger.info('Input Description : %s', data['description'])

    # Handle the processing of the data
    processor = Processor(data['description'])
    json_data = processor.process()

    response_data = {
        "message": "Success",
        "diagram_name": json_data['diagram_name'],
        "code_snippet": json_data['diagram_code'],
        "container_diagram_name": json_data['container_diagram_name'],
        "container_code_snippet": json_data['container_code']
    }
    return jsonify(response_data)

# Rebuild the diagram based on the code snippet
@app.route('/api/code', methods=['POST'])
def rebuild_diagram():
    data = request.get_json()
    app.logger.info('Rebuild Diagram : %s', data['code_snippet'])

    diagram_generator = DiagramGenerator()
    json_data = diagram_generator.parse_text_to_code(data['code_snippet'])
    diagram_name = diagram_generator.generate_diagram_json(json_data)

    response_data = {
        "message": "Success",
        "diagram_name": diagram_name,
        "code_snippet": data['code_snippet']
    }
    return jsonify(response_data)

# Get generated diagram
@app.route('/outputs/<path:filename>')
def serve_image(filename):
    print(filename)
    return send_from_directory('outputs', filename)

if __name__ == '__main__':
    app.run(debug=True)
