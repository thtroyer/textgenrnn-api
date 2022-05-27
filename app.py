import os.path

from flask import Flask, request
from textgenrnn import textgenrnn

app = Flask(__name__)


@app.route('/generate', methods=['GET'])
def prompt():
    global interactive

    args = request.args
    prompt = args.get('prompt')

    result = interactive.get_generatated_text(prompt)
    return result


class InteractiveTextGen:
    def __init__(self, model_path):
        self.textgen = self.load_model(model_path)

    def load_model(self, model: str):
        if not os.path.isfile(model):
            raise Exception(f"Can't find model {model}")
        return textgenrnn(model)

    def get_generatated_text(self, prompt: str) -> str:
        result = self.textgen.generate(1, temperature=1.0, prefix=prompt, return_as_list=True)
        return result[0]


interactive = InteractiveTextGen("./model_50.hdf5")

if __name__ == '__main__':
    app.run()
