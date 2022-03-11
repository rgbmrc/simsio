from keras.models import model_from_json

from simsio.serializers import TxtSerializer


class KerasModelSerializer(TxtSerializer):
    typ = "t"
    ext = ".json"

    def __init__(self):
        super().__init__()

    def load(self, f):
        return model_from_json(super().load(f))

    def dump(self, d, f):
        super().dump(d.to_json(), f)


class KerasWeightSerializer:
    typ = "b"
    ext = ".h5"

    @staticmethod
    def load(f):
        return f

    @staticmethod
    def dump(d, f):
        d.save_weights(f.name)
