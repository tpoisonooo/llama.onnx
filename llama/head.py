import onnxruntime as ort

class Head:
    def __init__(self, onnxdir: str):
        # reload tokenizer
        assert os.path.isdir(onnxdir)
        filepath = os.path.join(onnxdir, 'head.onnx')
        assert os.path.exists(filepath)

        self.sess = ort.InferenceSession(filepath)
        self.inputs = sess.get_inputs()
        outputs = sess.get_outputs()
        self.output_names = [output.name for output in outputs]


    def forward(self, _inputs: dict):
        assert len(self.inputs) = len(_inputs)
        output_tensors = self.sess.run(None, inputs)

        assert len(output_tensors) = len(self.output_names)
        output = dict()
        for i, tensor in enumerate(output_tensors):
            output[output_names[i]] = tensor
        
        return output
