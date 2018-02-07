/*
 * deep_fm_serving_client.cpp
 *
 *  Created on: 2018年2月7日
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
    //int64_t iBegin = TNOWMS;
    //int64_t iEnd   = TNOWMS;

    predictRequest.mutable_model_spec()->set_name(model_name);
    predictRequest.mutable_model_spec()->set_signature_name(model_signature_name); //serving_default

    //iBegin = TNOWMS;
    google::protobuf::Map<tensorflow::string, tensorflow::TensorProto>& inputs = *predictRequest.mutable_inputs();

    //feature to tfrequest
    std::vector<long>  ids_vec = {1,2,3,4,5,6,7,8,9,10,11,12,13,15,555,1078,17797,26190,26341,28570,35361,35613,
    		35984,48424,51364,64053,65964,66206,71628,84088,84119,86889,88280,88283,100288,100300,102447,109932,111823};
    std::vector<float> vals_vec = {0.05,0.006633,0.05,0,0.021594,0.008,0.15,0.04,0.362,0.1,0.2,0,0.04,
    		1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    tensorflow::TensorProto feat_ids;
    for (uint32_t i = 0; i < ids_vec.size(); i++) {
    	feat_ids.add_int64_val(ids_vec[i]);
    }
    feat_ids.mutable_tensor_shape()->add_dim()->set_size(1);	//batch_size
    feat_ids.mutable_tensor_shape()->add_dim()->set_size(feat_ids.int64_val_size());
    feat_ids.set_dtype(tensorflow::DataType::DT_INT64);
    inputs["feat_ids"] = feat_ids;

    tensorflow::TensorProto feat_vals;
    for (uint32_t i = 0; i < vals_vec.size(); i++) {
    	feat_vals.add_float_val(vals_vec[i]);
    }
    feat_vals.mutable_tensor_shape()->add_dim()->set_size(1);	//batch_size
    feat_vals.mutable_tensor_shape()->add_dim()->set_size(feat_vals.float_val_size());	//sample size
    feat_vals.set_dtype(tensorflow::DataType::DT_FLOAT);
    inputs["feat_vals"] = feat_vals;

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


