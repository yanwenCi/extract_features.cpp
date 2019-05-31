# extract_features.cpp
extract features for U-Net using C++, modified codes based on extract_features.cpp
1. read a volume of 3d raw data
   
2. put the data array into memory_data_layer
add a new function AddRawData in memory_data_layer.cpp 

template <typename Dtype>
void MemoryDataLayer<Dtype>::AddRawData(float* mat_vector,
	    const vector<int>& labels, int num) {
	//size_t num = mat_vector.size();
	//size_t num = batch_size_;
	std::cout << "nSlice：" << num << endl;
	std::cout << "nchannels：" << channels_ << endl;
	std::cout << "nheight：" << height_ << endl;
	std::cout << "nwidth：" << width_ << endl;

	CHECK(!has_new_data_) <<
		"Can't add mat until current data has been consumed.";
	CHECK_GT(num, 0) << "There is no mat to add";
	CHECK_EQ(num % batch_size_, 0) <<
		"The added data must be a multiple of the batch size.";
	added_data_.Reshape(num, channels_, height_, width_);
	added_label_.Reshape(num, 1, 1, 1);
	// Apply data transformations (mirror, scale, crop...)
	//this->data_transformer_->Transform(mat_vector, &added_data_);// this layer doesn't need transform
	//Dtype* transformed_data = transformed_blob->mutable_cpu_data(); 


	Dtype* feature_blob_data = added_data_.mutable_cpu_data();

	for (int i = 0; i < num*channels_*height_*width_; i++){
	 feature_blob_data[i] = mat_vector[i];
	}

	//// save images for checking
	//FILE * fpFeatures = NULL;
	////fopen_s(&fpFeatures, "D:\\iyuzu\\UNet\\unet-master\\image.raw", "wb");
	//fopen_s(&fpFeatures, "feature_blob.raw", "wb+");//93-A2-PDW-mag2_new
	//if (fpFeatures)
	//{
	//	fwrite(feature_blob_data, sizeof(Dtype), 128 * 128 * 60, fpFeatures);
	//	fclose(fpFeatures);
	//}// save images for checking

	Dtype* top_label = added_label_.mutable_cpu_data();
	for (int item_id = 0; item_id < num; ++item_id) {
		//top_label[item_id] = labels[item_id];
		top_label[item_id] = labels[0];
	}
	// num_images == batch_size_
	Dtype* top_data = added_data_.mutable_cpu_data();
	Reset(top_data, top_label, num);
	has_new_data_ = true;

}
add declaration in memory_data_layer.hpp

  virtual void AddRawData(float* mat_vector,
  	const vector<int>& labels, int num);
  
  re-build caffe

3.re-construct the slices of output feature map into 3d raw data

4.compile the extract_features.cpp and invoke the exe
                        "This program takes in a trained network and an input data layer, and then"
			" extract features of the input data produced by the net.\n"
			"Usage: extract_features  input_image_names pretrained_net_param"
			"  feature_extraction_proto_file  extract_feature_blob_name1[,name2,...]"
			"  save_feature_dataset_name1[,name2,...]  num_mini_batches  input image path"
			"  [CPU/GPU] [DEVICE_ID=0]\n"
			"Note: you can extract multiple features in one pass by specifying"
			" multiple feature blob names and dataset names separated by ','."
			" The names cannot contain white space characters and the number of blobs"
			" and datasets must be equal.";
			
