import tensorflow as tf
import sys
import grpc
import logging

from proto_files import predict_pb2, prediction_service_pb2, model_pb2, model_pb2_grpc
from grpc import RpcError
from grpc.beta import implementations

logger = logging.getLogger(__name__)

class PredictClient():

    def __init__(self, host, port, model_name, model_version):
        # DEFINE A CLIENT GIVEN:
        #   HOST (LOCALHOST IP),
        #   PORT (DETERMINED WHEN SERVING),
        #   MODEL NAME (DETERMINED WHEN SERVING),
        #   MODEL VERSION (DETERMINED WHEN EXPORTING)
        self.host = host
        self.port = port
        self.model_name = model_name
        self.model_version = model_version

    def predict(self, request_data, signature_name='serving_default', request_timeout=10, shape=[1, 3]):

        logger.info('Sending request to tfserving model')
        logger.info('Model name: ' + str(self.model_name))
        logger.info('Model version: ' + str(self.model_version))
        logger.info('Host: ' + str(self.host))

        # CONVERT FROM REQUEST ARRAY TO TENSOR FOR USE WITH TENSORFLOW
        features_tensor_proto = tf.contrib.util.make_tensor_proto(request_data,
                                                                    dtype=tf.float32, shape=shape)

        # CREATE GRPC CLIENT AND REQUEST
        channel = implementations.insecure_channel(self.host, int(self.port))
        stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        request.model_spec.signature_name = signature_name

        # SANITY CHECK ON MODEL VERSION
        if self.model_version > 0:
            request.model_spec.version.value = self.model_version

        # ADD REQUEST TENSOR TO REQUEST
        request.inputs['x'].CopyFrom(features_tensor_proto)

        # TRY TO EXECUTE THE REQUEST
        try:
            result = stub.Predict(request, timeout=request_timeout)
            return list(result.outputs['prediction'].int64_val)
        except RpcError as e:
            logger.error(e)
            logger.error('Prediction failed!')