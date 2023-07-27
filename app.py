#!/usr/bin/env python3.8
import argparse
from flask import Flask, render_template, request
from onnx_modifier import onnxModifier

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/open_model', methods=['POST'])
def open_model():
    # https://blog.miguelgrinberg.com/post/handling-file-uploads-with-flask
    onnx_file = request.files['file']
    global onnx_modifier
    onnx_modifier = onnxModifier.from_name_stream(onnx_file.filename, onnx_file.stream)
    print(onnx_file.name, onnx_file.stream)
    return 'OK', 200

# user start 

def read_temporary_file_to_list(temp_file):
    temp_file.seek(0)  # 파일 포인터를 파일의 처음으로 이동
    #content_list = temp_file.readlines()  # 파일의 내용을 리스트로 읽기
    content_list = [line.strip() for line in temp_file]
    temp_file.close()
    return content_list

@app.route('/open_text', methods=['POST'])
def open_text():
    #print("hello")
    text_file = request.files['file']
    #print(text_file.name, text_file.stream)
    content_list = read_temporary_file_to_list(text_file)

    #global onnx_modifier
    #text_plan = onnxModifier.from_name_stream(text_file.filename, text_file.stream)
    #print(text_file.name, text_file.stream)
    return 'OK', 200

@app.route('/download_text', methods=['POST'])
def text_and_download_model():
    modify_info = request.get_json()    

    print(modify_info)
    
    return 'OK', 200

# user end 


@app.route('/download', methods=['POST'])
def modify_and_download_model():
    modify_info = request.get_json()    

    print(modify_info)
    onnx_modifier.reload()   # allow downloading for multiple times
    onnx_modifier.modify(modify_info)
    onnx_modifier.check_and_save_model()

    return 'OK', 200

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='127.0.0.1', help='the hostname to listen on. Set this to "0.0.0.0" to have the server available externally as well')
    parser.add_argument('--port', type=int, default=5003, help='the port of the webserver. Defaults to 5000.')
    parser.add_argument('--debug', type=bool, default=False, help='enable or disable debug mode.')
    
    args = parser.parse_args()
    return args
    
def main():
    args = parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()
