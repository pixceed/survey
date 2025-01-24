from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context, send_file
from flask_cors import CORS
import requests



app = Flask(__name__)
CORS(app)


# test API
# curl -X POST -H "Content-Type: application/json" -d '{"key": "value"}' http://127.0.0.1:6601/echo
@app.route('/echo', methods=['POST'])
def echo():
    data = request.get_json()
    return jsonify(data)




########################################################################
# メイン
########################################################################
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6601, debug=True)