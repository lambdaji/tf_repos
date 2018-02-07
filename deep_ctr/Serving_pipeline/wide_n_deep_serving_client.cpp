/*
 * wide_n_deep_serving_client.cpp
 *
 *  Created on: 2017��10��28��
 *      Author: lambdaji
 */

#include "wide_n_deep_serving_client.h"
#include "google/protobuf/map.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"
#include <vector>
#include <string>

typedef google::protobuf::Map<tensorflow::string, tensorflow::TensorProto> OutMap;

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;

int ServingClient::callPredict(const std::string& model_name, const std::string& model_signature_name, std::map<std::string, std::string> & result)
{
    PredictRequest predictRequest;
    PredictResponse response;
    ClientContext context;
    tensorflow::TensorProto req_tp;
    tensorflow::Example example;
    //int64_t iBegin = TNOWMS;
    //int64_t iEnd   = TNOWMS;

    predictRequest.mutable_model_spec()->set_name(model_name);
    predictRequest.mutable_model_spec()->set_signature_name(model_signature_name); //serving_default

    //iBegin = TNOWMS;
    google::protobuf::Map<tensorflow::string, tensorflow::TensorProto>& inputs = *predictRequest.mutable_inputs();
    google::protobuf::Map<tensorflow::string, tensorflow::Feature>& feature_dict = *example.mutable_features()->mutable_feature();

    //feature to example
    std::vector<float> C_COLUMNS(13, 0.5);
    std::vector<long>  D_COLUMNS(26, 123);
    for(uint32_t i = 0; i < C_COLUMNS.size(); i++)
    	feature_dict["I" + std::to_string(i+1)].mutable_float_list()->add_value(C_COLUMNS[i]);
    for(uint32_t i = 0; i < D_COLUMNS.size(); i++)
    	feature_dict["C" + std::to_string(i+1)].mutable_int64_list()->add_value(D_COLUMNS[i]);

    //serialize to req.inputs
    std::string serialized;
    example.SerializeToString(&serialized);
    req_tp.mutable_tensor_shape()->add_dim()->set_size(1);		//set_size(5) for batch predicting
    req_tp.set_dtype(tensorflow::DataType::DT_STRING);
    req_tp.add_string_val(serialized);		//1st
    //req_tp.add_string_val(serialized);	//2nd
    //req_tp.add_string_val(serialized);	//3rd
    //req_tp.add_string_val(serialized);	//4th
    //req_tp.add_string_val(serialized);	//5th
    inputs["inputs"] = req_tp;

    //iEnd   = TNOWMS;
    //TLOGDEBUG("ServingClient::callPredict sample_to_tfrequest timecost(ms):" << (iEnd - iBegin) << endl);

    //predict
    //iBegin = TNOWMS;
    Status status = _stub->Predict(&context, predictRequest, &response);
    //iEnd   = TNOWMS;
    //TLOGDEBUG("ServingClient::callPredict _stub->Predict timecost(ms):" << (iEnd - iBegin) << endl);

    if (status.ok())
    {
        //TLOGDEBUG("ServingClient::callPredict call predict ok" << endl);
        //TLOGDEBUG("ServingClient::callPredict outputs size is " << response.outputs_size() << endl);

        OutMap& map_outputs = *response.mutable_outputs();
        OutMap::iterator iter;
        int output_index = 0;

        for (iter = map_outputs.begin(); iter != map_outputs.end(); ++iter)
        {
            tensorflow::TensorProto& result_tensor_proto = iter->second;
            tensorflow::Tensor tensor;
            bool converted = tensor.FromProto(result_tensor_proto);
            if (converted)
            {
                //TLOGDEBUG("ServingClient::callPredict the result tensor[" << output_index << "] is:" << tensor.SummarizeValue(10) << endl);
                result[iter->first] = tensor.SummarizeValue(10);
            }
            else
            {
                //TLOGDEBUG("ServingClient::callPredict the result tensor[" << output_index << "] convert failed." << endl);
            }
            ++output_index;
        }

        return 0;
    }
    else
    {
        //TLOGDEBUG("ServingClient::callPredict gRPC call return code: " << status.error_code() << ": " << status.error_message() << endl);
        return -1;
    }
}
