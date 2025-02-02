
#include "SparseSDMemory.h"
#include <assert.h>
#include <iostream>
#include "utility.h"

SparseSDMemory::SparseSDMemory(id_t id, std::string name, Config stonne_cfg, Connection* write_connection) : MemoryController(id, name) {
    this->write_connection = write_connection;
    //Collecting parameters from the configuration file
    this->num_ms = stonne_cfg.m_MSNetworkCfg.ms_size;  //Used to send data
    this->n_read_ports=stonne_cfg.m_SDMemoryCfg.n_read_ports;
    this->n_write_ports=stonne_cfg.m_SDMemoryCfg.n_write_ports;
    this->write_buffer_capacity=stonne_cfg.m_SDMemoryCfg.write_buffer_capacity;
    this->port_width=stonne_cfg.m_SDMemoryCfg.port_width;
    //End collecting parameters from the configuration file
    //Initializing parameters
    this->ms_size_per_input_port = this->num_ms / this->n_read_ports;
    this->write_fifo = new Fifo(write_buffer_capacity);
    for(int i=0; i<this->n_read_ports; i++) {
        Fifo* read_fi = new Fifo(this->write_buffer_capacity);
        Fifo* psum_fi = new Fifo(this->write_buffer_capacity);
        input_fifos.push_back(read_fi);
        psum_fifos.push_back(psum_fi);
        this->sdmemoryStats.n_SRAM_read_ports_weights_use.push_back(0);  //To track information
        this->sdmemoryStats.n_SRAM_read_ports_inputs_use.push_back(0);   //To track information
        this->sdmemoryStats.n_SRAM_read_ports_psums_use.push_back(0);    //To track information
    }
    for(int i=0; i<this->n_write_ports; i++) {  //To track information
        this->sdmemoryStats.n_SRAM_write_ports_use.push_back(0);  //To track information
    }  //To track information
    this->configuration_done = false;
    this->stationary_distributed = false;
    this->stationary_finished = false;
    this->stream_finished = false;
    this->execution_finished = false;
    this->metadata_loaded = false;
    this->layer_loaded = false;
    this->local_cycle=0;
    this->str_current_index = 0;
    this->current_state = CONFIGURING;
    this->multiplier_dest_table = NULL;
    this->str_counters_table = NULL;
    this->current_output = 0;
    this->output_size = 0;
    this->start_column_index = 0; //Stationary matrix current index (e.g., row in MK)
    this->sta_current_index_matrix = 0;
    this->prev_weight = 0; //yujin:weight
    this->cur_weight = 0;
    this->sta_iter_completed = false;
    this->current_output_iteration = 0;
    this->output_size_iteration = 0;
    this->current_row_index = 0;
    this->last_row_next_start_index = 0;
    this->n_ones_sta_matrix=0;
    this->n_ones_str_matrix=0;
    this->last_count_column_index=0;
    this->last_count_column_index_weight = 0;

    this->prev_sta_last_j_metadata =0;
    this->prev_sta_last_j_metadata_weight =0;
}

SparseSDMemory::~SparseSDMemory() {
    delete write_fifo;
    //Deleting the input ports
    for(int i=0; i<this->n_read_ports; i++) {
        delete input_fifos[i];
        delete psum_fifos[i];
    }
    
    if(this->layer_loaded) {
        delete[] multiplier_dest_table;
	delete[] str_counters_table;
    }

}

void SparseSDMemory::setWriteConnections(std::vector<Connection*> write_port_connections) {
    this->write_port_connections=write_port_connections; //Copying all the poiners 
    //assert(this->write_port_connections.size()==this->n_write_ports); 
}

void SparseSDMemory::setReadConnections(std::vector<Connection*> read_connections) {
    assert(read_connections.size() == n_read_ports); //Checking that the number of input ports is valid.
    this->read_connections = read_connections; //Copying all the pointers
}

void SparseSDMemory::setLayer(DNNLayer* dnn_layer, address_t input_address, address_t filter_address, metadata_address_t mapping_table, address_t output_address, Dataflow dataflow) {
    this->dnn_layer = dnn_layer;
    this->mapping_table = mapping_table;
    assert(this->dnn_layer->get_layer_type()==GEMM);  // This controller only supports GEMM with sparsity
    this->dataflow = dataflow;

    this->output_address = output_address;
    this->layer_loaded = true;


    //Loading parameters according to the equivalence between CNN layer and GEMM. This is done
    //in this way to keep the same interface.
    this->M = this->dnn_layer->get_K();
    this->K = this->dnn_layer->get_S();   //Be careful. K in GEMMs (SIGMA taxonomy) is not the same as K in CNN taxonomy (number of filters)
    //mapping_table = metadata의 1개의 column
    //Be careful. K in GEMMs (SIGMA taxonomy) is not the same as K in CNN taxonomy (number of filters)
    this->N = this->dnn_layer->get_X();  //In this case both parameters match each other.
    this->R = this->dnn_layer->get_K(); //yujin: Number of mapping_table rows
    std::cout << "Value of M=" << M << std::endl;
    std::cout << "Value of N=" << N << std::endl;
    std::cout << "Value of K=" << K << std::endl;
    std::cout << "Value of R=" << R << std::endl;
    sdmemoryStats.dataflow=dataflow; 

    if(dataflow==MK_STA_KN_STR) {
	std::cout << "Running MK_STA_KN_STR using mapping_table Dataflow" << std::endl;
        this->STA_address = input_address;
	this->dim_sta = M;
	this->STR_address = filter_address;
	this->dim_str = N;

    
	//MK_sta_ KN STR dataflow. According to the distribution of the bitmap
        this->STA_DIST_ELEM=1;
        this->STA_DIST_VECTOR=K;

        this->STR_DIST_ELEM=dim_str;
        this->STR_DIST_VECTOR=1;

	    this->OUT_DIST_VN=dim_str;
        this->OUT_DIST_VN_ITERATION=1;
	 

    }

    else if(dataflow==MK_STR_KN_STA) {
	std::cout << "Running MK_STR_KN_STA Dataflow" << std::endl;
        this->STA_address = filter_address;
	this->dim_sta = N;
	this->STR_address = input_address;
	this->dim_str= M;

	this->STA_DIST_ELEM=dim_sta;
	this->STA_DIST_VECTOR=1;

	this->STR_DIST_ELEM=1;
	this->STR_DIST_VECTOR=K;

	this->OUT_DIST_VN=1;
    this->OUT_DIST_VN_ITERATION=dim_sta;



    }

    else {
        std::cout << "Dataflow not recognised" << std::endl;
	assert(false);
    }


    this->output_size = dim_sta*dim_str;
    this->multiplier_dest_table = new unsigned int[M*K*N]; //yujin: mapping table에서 1이 있는 위치에 순서대로 multiplier를 할당하는 table
    this->str_counters_table = new unsigned int[M*K*N];
}


void SparseSDMemory::setLayer(DNNLayer* dnn_layer, address_t MK_address, address_t KN_address, address_t output_address, Dataflow dataflow) {
    this->dnn_layer = dnn_layer;
    assert(this->dnn_layer->get_layer_type()==GEMM);  // This controller only supports GEMM with sparsity
    this->dataflow = dataflow; 

    this->output_address = output_address;
    this->layer_loaded = true;


    //Loading parameters according to the equivalence between CNN layer and GEMM. This is done
    //in this way to keep the same interface.
    this->M = this->dnn_layer->get_K();
    this->K = this->dnn_layer->get_S();   //Be careful. K in GEMMs (SIGMA taxonomy) is not the same as K in CNN taxonomy (number of filters)
    this->N = this->dnn_layer->get_X();  //In this case both parameters match each other.
    std::cout << "Value of M=" << M << std::endl;
    std::cout << "Value of N=" << N << std::endl;
    std::cout << "Value of K=" << K << std::endl;
    sdmemoryStats.dataflow=dataflow; 

    if(dataflow==MK_STA_KN_STR) {
	std::cout << "Running MK_STA_KN_STR Dataflow" << std::endl;
        this->STA_address = MK_address;
	this->dim_sta = M;
	this->STR_address = KN_address;
	this->dim_str = N;

    
	//MK_sta_ KN STR dataflow. According to the distribution of the bitmap
        this->STA_DIST_ELEM=1;
        this->STA_DIST_VECTOR=K;

        this->STR_DIST_ELEM=dim_str;
        this->STR_DIST_VECTOR=1;

	this->OUT_DIST_VN=dim_str;
        this->OUT_DIST_VN_ITERATION=1;
	 

    }

    else if(dataflow==MK_STR_KN_STA) {
	std::cout << "Running MK_STR_KN_STA Dataflow" << std::endl;
        this->STA_address = KN_address;
	this->dim_sta = N;
	this->STR_address = MK_address;
	this->dim_str= M;

	this->STA_DIST_ELEM=dim_sta;
	this->STA_DIST_VECTOR=1;

	this->STR_DIST_ELEM=1;
	this->STR_DIST_VECTOR=K;

	 this->OUT_DIST_VN=1;
     this->OUT_DIST_VN_ITERATION=dim_sta;



    }

    else {
        std::cout << "Dataflow not recognised" << std::endl;
	assert(false);
    }


    this->output_size = dim_sta*dim_str;
    this->multiplier_dest_table = new unsigned int[dim_sta*K];
    this->str_counters_table = new unsigned int[dim_str*K];

}


//Load bitmaps
void SparseSDMemory::setSparseMetadata(metadata_address_t MK_metadata, metadata_address_t KN_metadata, metadata_address_t output_metadata) {
    if(this->dataflow==MK_STA_KN_STR) {
        this->STA_metadata = MK_metadata;
        this->STR_metadata = KN_metadata;
        this->output_metadata = output_metadata;

    }

    else if(dataflow==MK_STR_KN_STA) {
        this->STA_metadata = KN_metadata;
	    this->STR_metadata = MK_metadata;
	    this->output_metadata = output_metadata;
    }

    else {
        std::cout << "Dataflow not recognised" << std::endl;
        assert(false);
    }

    this->metadata_loaded = true;

}



void SparseSDMemory::cycle() {
    //Sending input data over read_connection
    std::cout<<"[SPARASESDMEMORY CYCLE CHECKING] start"<<std::endl;
    std :: cout <<"[START mapping_table column index] = " << start_column_index << std::endl;
    assert(this->layer_loaded);  // Layer has been loaded


    //assert(this->metadata_loaded); //Metadata for sparsity has been loaded
    std::vector<DataPackage*> data_to_send; //Input and weight temporal storage
    std::vector<DataPackage*> psum_to_send; 

    this->local_cycle+=1;
    this->sdmemoryStats.total_cycles++; //To track information



    if(current_state==CONFIGURING) {   //If the architecture has not been configured
        std::cout<<"[SPARASESDMEMORY CYCLE CHECKING] CONFIGURING"<<std::endl;
        int i = start_column_index;  //start mapping_table column index

        int n_ms = 0; //yujin: Number of multipliers assigned
        int n_current_cluster = 0; //yujin: 1개의 column에서의 1의 개수 count-> sparseVN -> (나중에 adder tree에 VN size 전달)
        this->configurationVNs.clear();
        this->vnat_table.clear();

        /*
	    if(this->current_row_index > 0)  { // We are managing one cluster with folding
            std::cout<< "[current_row_index > 0] we are managing one cluster with folding"<<std::endl;
            n_ms++; //One for the psum
	        //row_index=this->current_row_index; //yujin: 중간에 끊긴 경우(= multiplier의 num < 1개의 column안에서 row index) 다음 row index부터 시작
	         while((n_ms < this->num_ms) && (current_row_index*M*K+i < M*K*4)&&(i<M*K)) { //yujin: R = mapping table의 row 수
                //TODO change MK if it is another dw
                //TODO YUJUIN : break if diff weight

                //if(prev_weight != cur_weight) {
                //cur_weight

                   // break;
                //}
                if(this->mapping_table[current_row_index * M*K + i]) { //yujin: STA_metadat는 mappind table에서 하나의 column을 의미
                    //yujin: i=current column index
                    //Add to the cluster
                    this->multiplier_dest_table[current_row_index * M*K + i]=n_ms; //DEST
                    std::cout<<"[CHECK MAPPING TABLE INDEX & MULTIPLIER DESTINATION] "<<"mapping_table index = "<<current_row_index * M*K+ i<< "multiplier destination"<< n_ms<<std::endl;
                    //yujin: 1이면, multiplier 할당
                    n_ms++; //yujin: 사용한 multiplier의 수 count
                    n_current_cluster++; 
                }
                current_row_index++; //yujin: 다음 row로 이동
	        }

        SparseVN VN(n_current_cluster, true); //Here folding is enabled and the SparseVN increments 1 to size
        //yujin: folding enable flag true -> 중간 row에서 부터 시작한 값
        this->configurationVNs.push_back(VN); //yujin: configurationVNs = A set of each VN size mapped onto the architecture
	    this->vnat_table.push_back(0); //Adding the current calculation (row or column) of this VN.

	    //Searching if there are more values in the next cluster. If not, update last_row_next_start_index to K to indicate that in the next iteration the next sta dim must be evaluated
	    int remaining_values = 0; //yujin: VN을 만들어 준 다음 row index부터 마지막 row index까지 1이 남아있는지 확인
	    for(int r=current_row_index; r<R; r++) {
                if(this->mapping_table[r * M*K + i]) {
                    remaining_values+=1;
	        }
	    }
	    if(remaining_values > 0) { //yujin: remaining_values가 있으면, j부터 k까지 다시 count해서 SparseVN을 만들어주어야 한다
	        this->last_row_next_start_index=current_row_index;
            //yujin: last_row_next_start_index = SparseVN을 만들어 준 다음 row의 index = 다음에 시작할 row index
	    }

	    else {
                this->last_row_next_start_index = R;
	    }
	}
    */

        //else { //yujin: folding이 아닌경우 = row index 0부터 시작하는 경우 (current_row_index = 0)
        while ((n_ms < this->num_ms) && (current_row_index * M * K + i < M * K * N) &&(i < M * K)) { //TODO change MK if it is another dw
            //yujin: i<M*K 조건 추가
            //TODO YUJUIN : break if diff weight

            if (this->mapping_table[current_row_index * M * K +i]) { //If the bit is enabled in the stationary bitmap
                //Add to the cluster
                this->multiplier_dest_table[current_row_index * M * K + i] = n_ms; //DEST
                std::cout<< "column index = "<< i << "row_index = "<<current_row_index<<std::endl;
                std::cout << "[CHECK MAPPING TABLE INDEX & MULTIPLIER DESTINATION] " << " mapping_table index = " << current_row_index * M * K + i << " multiplier destination" << n_ms << std::endl;
                n_ms++;
                n_current_cluster++;
            }

            prev_weight = STR_address[i];
            //std::cout << "current row index = " << current_row_index << std::endl;
            current_row_index++; // Next elem in vector (end next row index)
            //std::cout << "next row index = " << current_row_index << std::endl;
            if (n_current_cluster > 0) {
                //Creating the cluster for this row
                SparseVN VN(n_current_cluster, false);
                this->configurationVNs.push_back(VN); //Adding to the list
                this->vnat_table.push_back(0); //Adding the current calculation (row or column) of this VN.
                n_current_cluster = 0;
            }

            if (current_row_index == R) { //yujin: row index = R인 경우에 SparseVN을 만들어주고 있음
                //Change cluster since we change of vector
                current_row_index = 0; //row index = 0
                //std::cout << "[CHECK ROW INDEX & COLUMN INDEX] " << " mapping_table column index =" << i << " row index"<< current_row_index << std::endl;
                i++; // Next column

                cur_weight = STR_address[i];
                /*
                if (n_current_cluster > 0) {
                    //Creating the cluster for this row
                    SparseVN VN(n_current_cluster, false);
                    this->configurationVNs.push_back(VN); //Adding to the list
                    this->vnat_table.push_back(0); //Adding the current calculation (row or column) of this VN.
                    n_current_cluster = 0;
                }
                */
                if(prev_weight != cur_weight)
                    break;
            }
        }

        //prev_sta_last_j_metadata = this->last_row_next_start_index;
        prev_sta_last_j_metadata = this->last_row_next_start_index;
        prev_sta_last_j_metadata_weight = this->last_row_next_start_index;


        //std::cout << "[CHECK START VN COLUMN_INDEX] VN start row index" << prev_sta_last_j_metadata <<std::endl;
        if (current_row_index < R) {
            //Find if there is a last cluster
            /*
            if (n_current_cluster > 0) {
                //Creating the cluster for this row
                SparseVN VN(n_current_cluster, false);
                this->configurationVNs.push_back(VN); //Adding to the list
                this->vnat_table.push_back(0); //Adding the current calculation (row or column) of this VN.
                n_current_cluster = 0;
            }
            */
            int remaining_values = 0;
            for (int r = current_row_index; r < R; r++) {
                if (this->mapping_table[r * M * K + i]) {
                    remaining_values += 1;
                }
            }
            if (remaining_values > 0) {
                this->last_row_next_start_index = current_row_index; // yujin: if is value, start from next value
            } else {                     
                this->last_row_next_start_index = R;
            }
        }

            /*
            if(this->configurationVNs.size()==0) { //If any entire cluster fits, then folding is needed to manage this cluster

                SparseVN VN(n_current_cluster, false); //Here folding is still disabled as this is the first iteration
                this->configurationVNs.push_back(VN);
                this->vnat_table.push_back(0); //Adding the current calculation (row or column) of this VN.
                       //Searching if there are more values in the next cluster. If not, update last_row_next_start_index to K to indicate that in the next iteration the next sta dim must be evaluated
                int remaining_values = 0;
                for(int r=current_row_index; r<R; r++) {
                    if(this->mapping_table[current_row_index * M*K + i]) {
                        remaining_values+=1;
                    }
                }
                if(remaining_values > 0) {
                    this->last_row_next_start_index=current_row_index;
                }

                 else {
                    this->last_row_next_start_index = R;
                }
            }
            */
/*
        else { //If there is at least one cluster, then all of them has size K and it is necessary to stream K
            //K elements
            int remaining_values = 0;
            for (int r = current_row_index; r < R; r++) {
                if (this->mapping_table[r * M * K + i]) {
                    remaining_values += 1;
                }
            }
            if (remaining_values > 0) {
                this->last_row_next_start_index = current_row_index;
            }
            else {
                this->last_row_next_start_index = R;
            }
        }
        */
        count_column_index = i; // save end mapping_table column index

        //std::cout << "[CHECK END NEXT START ROW INDEX] end row index" << last_row_next_start_index-1 <<std::endl;
        //std::cout << "[CHECK END VN COLUMN_INDEX] VN end column index" << count_column_index <<std::endl;

        // } //end else whole rows


        //Calculating the STR SOURCE TABLE with the indexes of each value

        //Once the VNs has been selected, lets configure the RN and MN
        // Configuring the multiplier network
        if (this->configurationVNs.size() == 0) { //yujin: configurationVNs에 아무것도 들어 있지 않은 경우
           std::cout << "Cluster size exceeds the number of multipliers in column " << this->start_column_index << std::endl;
           //assert(false);
        }

        for (int i = 0; i < this->configurationVNs.size(); i++) {
            std::cout << "[CHECK VN SIZE USING MAPPING TABLE] VN size = " << this->configurationVNs[i].get_VN_Size()<< std::endl;
        }
        this->sdmemoryStats.n_sta_vectors_at_once_avg += this->configurationVNs.size(); //accumul
        if (this->configurationVNs.size() > this->sdmemoryStats.n_sta_vectors_at_once_max) {
            this->sdmemoryStats.n_sta_vectors_at_once_max = this->configurationVNs.size();
        }

        this->sdmemoryStats.n_reconfigurations++;

        std::cout << "Configuring the MULTIPLIER & REDUCTION Networks" << std::endl;
        this->multiplier_network->resetSignals(); //Reseting the values to default
        this->multiplier_network->configureSparseSignals(this->configurationVNs, this->dnn_layer, this->num_ms);
        //Configuring the reduce network
        this->reduce_network->resetSignals(); //Reseting the values to default
        this->reduce_network->configureSparseSignals(this->configurationVNs, this->dnn_layer, this->num_ms);
        std::cout << "End Networks configuring" << std::endl;
        //yujin: Number of psums to calculate in this iteration
        //this->output_size_iteration = this->configurationVNs.size();
    }



    else if(current_state == DIST_STA_MATRIX) {
        //int address_offset = start_column_index;
        //Distribution of the stationary matrix
        std::cout<<"[SPARASESDMEMORY CYCLE CHECKING] DIST_STA_MATRIX"<<std::endl;
        //std::cout<<"[start_column_index]"<<start_column_index<<std::endl;
        unsigned int dest = 0; //MS destination
        int row_size = this->R;
        //unsigned int sub_address = 0;
        int prev_last_count_column_index_weight = last_count_column_index_weight;

        for(int i=last_count_column_index_weight; i<=this->count_column_index; i++) {
            if (i == M * K)
                break;
            if(prev_sta_last_j_metadata_weight==R){
                prev_sta_last_j_metadata_weight = 0; //row_index
            }
            if (i == count_column_index) {
                row_size = this->last_row_next_start_index; // last 1 idx
            }
            /*
	        if(this->configurationVNs[i].getFolding()) {
             j=1;
	        dest++; //Avoid the one in charge of the psum
	        }
            */
            //std::cout<<"[address_offset]"<<address_offset<<std::endl;
            int last_j = prev_sta_last_j_metadata_weight;
            for(int j = prev_sta_last_j_metadata_weight; j<row_size; j++) {
	            //Accessing to memory
                if (mapping_table[j * M * K + i]) {
                    data_t data = this->STR_address[i]; //yujin: WEIGHT_MATRIX[column index]
                    //prev_weight =data;
                    //std::cout<< "[PREV_WEIGHT] prev_weight value is : "<<prev_weight <<std::endl;

                    sdmemoryStats.n_SRAM_weight_reads++;
                    this->n_ones_sta_matrix++;
                    //std::cout<<"!!!!!!!!!!!! dest = "<< dest<<std::endl;
                    DataPackage *pck_to_send = new DataPackage(sizeof(data_t), data, WEIGHT, 0, UNICAST, dest);
                    //std::cout<<"dest, data = " << dest <<data<<std::endl;
                    this->sendPackageToInputFifos(pck_to_send);
                    dest++;
                    //sub_address++;
                }
                last_j = j + 1;
	        }
            if(last_j==R) {
                prev_last_count_column_index_weight = i + 1;
                prev_sta_last_j_metadata_weight = 0;
            }
            //address_offset++;
        }
        last_count_column_index_weight = prev_last_count_column_index_weight;
    }

/*
    else if(current_state == DIST_STR_MATRIX) {
        std::cout<< "DIST_STR_MATRIX"<<start_column_index<<std::endl;
        int address_offset;
        //yujin: make input index (use data)
        if(this->current_row_index > 0)  {
            for (int i = current_row_index; i <= start_column_index + this->configurationVNs.size(); i++) {
                for (int j = 0 ; j < this->R; j++) {
                    if (mapping_table[j * M*K + i] && (j < this->configurationVNs.size())) {
                        str_counters_table[j * M*K + i] = STA_address[address_offset];
                        std::cout << "test counter table" << " nonzero mapping table index check :" << j * M*K + i <<" / data :" << STR_address[address_offset] <<std::endl;
                    }
                }
            }
        }

        else {
            for (int i = start_column_index; i <= start_column_index + this->configurationVNs.size(); i++) {
                for (int j = 0; j < this->R; j++) {
                    if (mapping_table[j * M*K + i] ) {
                        str_counters_table[j * M * K + i] = j;
                        std::cout << "test counter table" << " nonzero mapping table index check :" << j * M*K + i <<" / data :" << STR_address[j] <<std::endl;
                    }
                }
            }
        }
*/

    else if(current_state == DIST_STR_MATRIX) {
        std::cout<<"[SPARASESDMEMORY CYCLE CHECKING] DIST_STR_MATRIX"<<std::endl;

        //std::cout << "last_count_column_index : " << last_count_column_index <<std::endl;
        //std::cout << "column index : " << count_column_index << std::endl;
        //yujin: make input index (use data)

        //std::cout << "current_row_index : " << current_row_index << std::endl;

        /*
        if(this->current_row_index > 0)  {
            int prev_last_count_column_index = last_count_column_index;
            int row_size = this->R;
            std::cout<<"last_row_next_start_index"<<this->last_row_next_start_index<<std::endl;
            for (int i = last_count_column_index; i <= count_column_index; i++) {
                if(i==M*K)
                    break;
                if(i==count_column_index){
                    row_size = this->last_row_next_start_index;
                }
                for (int j = 0 ; j < row_size; j++) {
                    if (mapping_table[j * M*K + i]) {
                        str_counters_table[j * M*K + i] = j;
                        std::cout << "test counter table" << " nonzero mapping table index check :" << j * M*K + i <<" / data :" << j <<std::endl;
                    }
                }
                prev_last_count_column_index = i+1;
            }
            last_count_column_index = prev_last_count_column_index;
        }
         */

        //else {
        int prev_last_count_column_index = last_count_column_index;
        int row_size = this->R;

        for (int i = last_count_column_index; i <= this->count_column_index; i++) {
            data_t data;
            if (i == M * K)
                break;
            if(prev_sta_last_j_metadata==R){
                prev_sta_last_j_metadata = 0; //row_index
            }
            if (i == count_column_index) {
                row_size = this->last_row_next_start_index; // last 1 idx
            }

            // start = prev_last , end = current last
            int last_j = prev_sta_last_j_metadata;
            for (int j = prev_sta_last_j_metadata; j < row_size; j++) {
                if (mapping_table[j* M * K + i]) {
                    str_counters_table[j * M * K + i] = j;
                    //std::cout<< "data row index -> j ="<< j <<std::endl;
                    std::cout << "test counter table" << " nonzero mapping table index check :" << j * M * K + i << " / data :" << j << std::endl;

                    unsigned int dest = multiplier_dest_table[j * M * K + i];
                    unsigned int src = str_counters_table[j * M * K + i];
                    data = STA_address[src];
                    //std::cout << "data" << STA_address[src] << std::endl;
                    sdmemoryStats.n_SRAM_input_reads++;

                    DataPackage *pck = new DataPackage(sizeof(data_t), data, IACTIVATION, 0, UNICAST, dest); //+yujin: save row index -> use output address index
                    this->sendPackageToInputFifos(pck);

                }
                else {
                    data = 0.0; //If the STA matrix has a value then the STR matrix must be sent even
                }
                last_j = j + 1;
            }
            if(last_j==R) {
                prev_last_count_column_index = i + 1;
                prev_sta_last_j_metadata = 0;
            }

        }
        std::cout << "prev_sta_last_j_metadata" << prev_sta_last_j_metadata << std::endl;
        std::cout << "last_row_next_start_index" << this->last_row_next_start_index << std::endl;
        last_count_column_index = prev_last_count_column_index;
        //}


        //int init_point_str = this->start_column_index;
        //std::cout<< "init_point_str" << init_point_str<<std::endl;
         //int end_point_str = this->count_column_index;
        //std::cout<< "end_point_str" << end_point_str<<std::endl;

        /*
        if (this->current_row_index > 0) { //If folding is enabled there is just a row on  fly]
            assert(this->configurationVNs.size() == 1);
            //send psum
            unsigned int addr_offset = start_column_index;
            bool *destinations = new bool[this->num_ms];
            for (int i = 0; i < this->num_ms; i++) {
                destinations[i] = false;
            }

            destinations[0] = true;
            //std::cout<< "addr_offset: "<<addr_offset<<std::endl;
            //std::cout << addr_offset<<std::endl;

            data_t psum = this->output_address[addr_offset];  //Reading the current psum
            std::cout << "output_address[addr_offset]: " << addr_offset << std::endl;
            std::cout << output_address[addr_offset] << std::endl;
            DataPackage *pck = new DataPackage(sizeof(data_t), psum, PSUM, 0, MULTICAST, destinations, this->num_ms);
            this->sdmemoryStats.n_SRAM_psum_reads++; //To track information
            this->sendPackageToInputFifos(pck);
        }
        */
        /*
        for (int i = init_point_str; i <= end_point_str; i++) {   //For each element in the current vector in the str matrix
            //Creating the bit vector for this value
            data_t data;
            if (i == M * K)
                break;
            if(prev_sta_last_j_metadata==R){
                prev_sta_last_j_metadata = 0; //row_index
            }
            if (i == count_column_index) {
                row_size = this->last_row_next_start_index; // last 1 idx
            }

            for (int j = 0; j < this->R; j++) {
                if (mapping_table[j * M * K + i]) {
                    unsigned int dest = multiplier_dest_table[j * M * K + i];
                    unsigned int src = str_counters_table[j * M * K + i];
                    data = STA_address[src];
                    std::cout << "data" << STA_address[src] << std::endl;
                    sdmemoryStats.n_SRAM_input_reads++;

                    DataPackage *pck = new DataPackage(sizeof(data_t), data, IACTIVATION, 0, UNICAST, dest); //+yujin: save row index -> use output address index
                    this->sendPackageToInputFifos(pck);
                    //std::cout<<"sendPackageToInputFifos"<<std::endl;
                }
                else {
                    data = 0.0; //If the STA matrix has a value then the STR matrix must be sent even
                }
            }
        }
         */
        //str_current_index++;
    }


    //Receiving output data from write_connection
    //this->receive();
    else if (current_state == WAITING_FOR_NEXT_STA_ITER){
        sta_iter_completed = true;
    }




/*
    if(!write_fifo->isEmpty()) {
        std::cout<<"[WRITE IFO] pop write_fifo -> write output_address (psum or ofmap)"<<std::endl;
        //Index the data by using the VN Address Table and the VN id of the packages
        for(int i=0; i<write_fifo->size(); i++) {
            DataPackage* pck_received = write_fifo->pop();
            //unsigned int vn = pck_received->get_vn();
            data_t data = pck_received->get_data();
            this->sdmemoryStats.n_SRAM_psum_writes++; //To track information
	        unsigned int addr_offset = start_column_index;//+vn)*OUT_DIST_VN + vnat_table[vn]*OUT_DIST_VN_ITERATION;
            // yujin: error! : edit start_column_index -> input pck row_index

	        //vnat_table[vn]++;
            this->output_address[addr_offset]=data; //ofmap or psum, it does not matter.
            current_output++;
	        current_output_iteration++;
            std::cout << "[COUNT COMPLETE FLAG] CURRENT_OUTPUT_ITERATION = " << current_output_iteration << std::endl;
	        if(write_fifo->isEmpty()) {
                current_output_iteration = 0;
                //sta_iter_completed=true;
	        }
            //std::cout << "write fifo i = " << i << std::endl;
            //std::cout << "write fifo size = " << write_fifo->size() << std::endl;
            delete pck_received; //Deleting the current package
        }
    }
*/
    //Transitions
    if(current_state==CONFIGURING) {
        if(this->configurationVNs.size()== 0) {
            //std::cout << "config vn size = " <<this->configurationVNs.size() << std::endl;
            start_column_index = count_column_index;
            if(start_column_index < M*K) {
                current_state = CONFIGURING;
            }
            else{
                this->execution_finished = true;
                current_state = ALL_DATA_SENT;
            }
        }
        else
            current_state=DIST_STA_MATRIX;

    }

    else if(current_state==DIST_STA_MATRIX) {
        current_state=DIST_STR_MATRIX;
    }

    else if(current_state==DIST_STR_MATRIX ){ //} && str_current_index==this->configurationVNs.size()) {
    //std::cout<<"[PSUM COUNT] psum complete (" << str_current_index << "/" << this->configurationVNs.size() << ")" << std::endl;
	current_state = WAITING_FOR_NEXT_STA_ITER;
    }
    else if (current_state == WAITING_FOR_NEXT_STA_ITER){
        current_state = FINAL_STATE;
    }

    else if(current_state==FINAL_STATE) {
    
	    //this->str_current_index = 0;
	    this->sta_iter_completed=false;
        //if(this->configurationVNs.size()==1) {//If there is only one VN, then maybe foliding has been needed
        this->current_row_index=this->last_row_next_start_index;
	   // if(this->configurationVNs[0].getFolding()) {
           //this->current_row_index-=1;
	   //}
        // yujin : next column
        // current_row_index : current row
        // start_column_index : next column
        // count_column_index : current column
	    if(this->current_row_index == this->R) { //If this is the end of the cluster, it might start to the next
            std::cout<< "row_index == R : go to next column"<<std::endl;
            //this->start_column_index+=1;
            this->start_column_index = count_column_index + 1;
		    this->current_row_index = 0;
        }
	//}
	    else { // yujin : remain column
            std::cout<< "row_index != R : remain current column"<<std::endl;
	        this->start_column_index = this->count_column_index;
	        std::cout << "STONNE: VN complete num (" << this->start_column_index << ")" << std::endl;
            //this->current_row_index = 0;
	    }

	    unsigned int total_size = 0;
        for(int i=0; i<this->configurationVNs.size(); i++) {
            total_size++;
	        if(this->configurationVNs[i].getFolding()) {
                total_size-=1; //Sustract the -1 of the extra multiplier
	        }
        }
	    this->sta_current_index_matrix+=total_size;
	    if((start_column_index>M*K-1)) {
            //yujin: this->configurationVNs.size() -> M*K*4
	        //Calculating sparsity values  and some final stats
	        //unsigned int sta_metadata_size = this->dim_sta*K;
	        //unsigned int str_metadata_size = this->dim_str*K;
	        //unsigned int sta_zeros = sta_metadata_size - this->n_ones_sta_matrix;
	        //unsigned int str_zeros = str_metadata_size - this->n_ones_str_matrix;
            //sdmemoryStats.sta_sparsity=(counter_t)((100*sta_zeros) / sta_metadata_size);
	        //sdmemoryStats.str_sparsity=(counter_t)((100*str_zeros) / str_metadata_size);
	        this->sdmemoryStats.n_sta_vectors_at_once_avg = this->sdmemoryStats.n_sta_vectors_at_once_avg / this->sdmemoryStats.n_reconfigurations;
            this->execution_finished = true; //if the last sta cluster has already be calculated then finish the sim
	        current_state = ALL_DATA_SENT;
	    }
	    else {
            current_state=CONFIGURING;
	    }
    }

    this->send();
}

bool SparseSDMemory::isExecutionFinished() {
    return this->execution_finished;
}

/* The traffic generation algorithm generates a package that contains a destination for all the ms. We have to divide it into smaller groups of ms since they are divided into several ports */
void SparseSDMemory::sendPackageToInputFifos(DataPackage* pck) {
    // BROADCAST PACKAGE
    if(pck->isBroadcast()) {
        //Send to all the ports with the flag broadcast enabled
        for(int i=0; i<this->n_read_ports; i++) {
            //Creating a replica of the package to be sent to each port
            DataPackage* pck_new = new DataPackage(pck->get_size_package(), pck->get_data(), pck->get_data_type(), i, BROADCAST); //Size, data, data_type, source (port in this case), BROADCAST
            //Sending the replica to the suitable fifo that correspond with the port
            if(pck->get_data_type() == PSUM) { //Actually a PSUM cannot be broadcast. But we put this for compatibility
                psum_fifos[i]->push(pck_new);
            }          
            else {  //INPUT OR WEIGHT
                std::cout << "send package to input fifos" << std::endl;
                //Seting iteration of the package
                pck_new->setIterationK(
                        pck->getIterationK()); //Used to avoid sending packages from a certain iteration without performing the previous.
                input_fifos[i]->push(pck_new);
            }
        }
    }

    // UNICAST PACKAGE
    else if(pck->isUnicast()) {
        //We only have to send the weight to one port and change the destination to adapt it to the subgroup
        unsigned int dest = pck->get_unicast_dest(); //This is according to ALL the mswitches.
        std::cout<<"[sendPackageToInputFifos] : (UNICAST)"<<std::endl;
        //std::cout<<"@@@@@@@@ dest = "<<dest<<std::endl;
        unsigned int input_port = dest / this->ms_size_per_input_port;
        unsigned int local_dest = dest % this->ms_size_per_input_port;


        //Creating the package
        std::cout<<"[send package] "<<"pck size : "<<pck->get_size_package()<<" pck data :"<<pck->get_data()<<" pck data type :"<< pck->get_data_type() << " input port :"<<input_port<<" local dest : "<<local_dest<<std::endl;
        DataPackage* pck_new = new DataPackage(pck->get_size_package(), pck->get_data(), pck->get_data_type(), input_port, UNICAST, local_dest); //size, data, type, source (port), UNICAST, dest_local
        // DataPackage::DataPackage(size_t size_package, data_t data, operand_t data_type, id_t source,traffic_t traffic_type, unsigned int unicast_dest) :
        //Sending to the fifo corresponding with port input_port
        if(pck->get_data_type() == PSUM) { //Actually a PSUM cannot be broadcast. But we put this for compatibility
            psum_fifos[input_port]->push(pck_new);
        }
        else {  //INPUT OR WEIGHT
            input_fifos[input_port]->push(pck_new);
            pck_new->setIterationK(pck->getIterationK());
        }
    }

    //MULTICAST PACKAGE 
    else { //The package is multicast and then we have to send the package to several ports
        const bool* dest = pck->get_dests();  //One position for mswitch in all the msarray
        bool thereis_receiver;
        printf("paket \n");
        for(int i=0; i<this->n_read_ports; i++) { //Checking each port with size this->ms_size_per_input_port each. Total=ms_size
            //std::cout<<"n_read_ports"<<n_read_ports<<std::endl;
            unsigned int port_index = i*this->ms_size_per_input_port;
            thereis_receiver = false; // To know at the end if the group
            bool* local_dest = new bool[this->ms_size_per_input_port]; //Local destination array for the subtree corresponding with the port i
            for(int j=0; j<this->ms_size_per_input_port; j++) {  //For each ms in the group of the port i
                printf("paket2 \n");
                printf("%d \n", j);
                local_dest[j] = dest[port_index + j]; //Copying the subarray
                if(local_dest[j] == true) {
                    thereis_receiver=true; // To avoid iterating again to know whether the data have to be sent to the port or not.
                }
            }

            if(thereis_receiver) { //If this port have at least one ms to true then we send the data to this port i
                printf("paket3 \n");
                DataPackage* pck_new = new DataPackage(pck->get_size_package(), pck->get_data(), pck->get_data_type(), i, MULTICAST, local_dest, this->ms_size_per_input_port);
                if(pck->get_data_type() == PSUM) {
                    psum_fifos[i]->push(pck_new);
                }
 
                else {
                    pck_new->setIterationK(pck->getIterationK());
                    input_fifos[i]->push(pck_new);
                    printf("paket4 \n");
                }
            }
            else {
                delete[] local_dest; //If this vector is not sent we remove it.
            }
        }
    }

    delete pck; // We have created replicas of the package for the ports needed so we can delete this
} 

void SparseSDMemory::send() {
    //std::cout<<"[SPARSE_SDMEMORY SEND]"<<std::endl;
    //Iterating over each port and if there is data in its fifo we send it. We give priority to the psums
    for(int i=0; i<this->n_read_ports; i++) {
        //std::cout<<"n_read_ports"<<this->n_read_ports<<std::endl;
        //std::cout<<"[SEND] CEHCKING PSUM & INPUT FIFO" << i << "is empty ?" << this-> input_fifos[i]->isEmpty() <<std::endl;

        if(!this->psum_fifos[i]->isEmpty()) { //If there is something we may send data though the connection
            std::cout<<"psum_fifos" << i << "is Empty?" <<this->psum_fifos[i]->isEmpty()<<std::endl;
            std::cout<<"[SEND PSUM -> WRITE_CONNECTION]"<<std::endl;
            std::vector<DataPackage*> pck_to_send;
            DataPackage* pck = psum_fifos[i]->pop();
#ifdef DEBUG_MEM_INPUT
            std::cout << "[MEM_INPUT] Cycle " << local_cycle << ", Sending a psum through input port " << i  << std::endl;
#endif
            pck_to_send.push_back(pck);
            this->sdmemoryStats.n_SRAM_read_ports_psums_use[i]++; //To track information
            //Sending to the connection
            this->read_connections[i]->send(pck_to_send);
        }
        //If psums fifo is empty then input fifo is checked. If psum is not empty then else do not compute. Important this ELSE to give priority to the psums and do not send more than 1 pck
        else if(!this->input_fifos[i]->isEmpty()) {
            std::vector<DataPackage*> pck_to_send;
            //std::cout<<"[INPUT FIFO NOT EMPTY]"<<std::endl;
            //If the package belongs to a certain k iteration but the previous k-1 iteration has not finished the package is not sent
            DataPackage* pck = input_fifos[i]->front(); //Front because we are not sure if we hazve to send it.

           
            if(pck->get_data_type()==WEIGHT) {
                this->sdmemoryStats.n_SRAM_read_ports_weights_use[i]++; //To track information
#ifdef DEBUG_MEM_INPUT
                std::cout << "[MEM_INPUT] Cycle " << local_cycle << ", Sending a WEIGHT through input port " << i << std::endl;
#endif
            }  
            else {
                this->sdmemoryStats.n_SRAM_read_ports_inputs_use[i]++; //To track information

#ifdef DEBUG_MEM_INPUT
                std::cout << "[MEM_INPUT] Cycle " << local_cycle << ", Sending an INPUT ACTIVATION through input port " << i << std::endl;
#endif
            }

            pck_to_send.push_back(pck); //storing into the vector data type structure used in class Connection


            this->read_connections[i]->send(pck_to_send); //Sending the input or weight through the connection
         //   std::vector<DataPackage*> a;
         //   a=pck_to_send;
         //   this->read_connections[i]->send(a); //Sending the input or weight through the connection


            input_fifos[i]->pop(); //pulling from fifo

        }
    }
}

//TODO Remove this connection
void SparseSDMemory::receive() { //TODO control if there is no space in queue
    if(this->write_connection->existPendingData()) {
        std::vector<DataPackage*> data_received = write_connection->receive();
        //std::cout<<"[DATA RECEIVE] write_connection -> write_fifo push"<<std::endl;
        for(int i=0; i<data_received.size(); i++) {
            //std::cout<<"[DATA RECEIVE] write_connection -> write_fifo push"<<std::endl;
            write_fifo->push(data_received[i]);
        }
    }
    //std::cout<<"write_port_connectionswrite_port_connections.size = "<<write_port_connections.size()<<std::endl;
    for(int i=0; i<write_port_connections.size(); i++) { //For every write port
        //std::cout<<"i, write port connections size "<<i<<","<<write_port_connections.size()<<std::endl;
        //yujin: seg fault!
        //printf("data=");
        std::cout<<write_port_connections[i]->existPendingData()<<std::endl;
        if(write_port_connections[i]->existPendingData()) {
            std::cout<<"existPendingData = "<<write_port_connections[i]->existPendingData()<<std::endl;
            //std::cout<<"exists pending data1"<< std::endl;
            std::vector<DataPackage*> data_received = write_port_connections[i]->receive();
            //std::cout<< "vector receive1"<<std::endl;
             for(int i=0; i<data_received.size(); i++) {
                 //std::cout << "write fifo push1"<<std::endl;

                 write_fifo->push(data_received[i]);
             }
        }    
    }
}

void SparseSDMemory::printStats(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "\"SDMemoryStats\" : {" << std::endl; //TODO put ID
    this->sdmemoryStats.print(out, indent+IND_SIZE);
    out << ind(indent) << "}"; //Take care. Do not print endl here. This is parent responsability
}

void SparseSDMemory::printEnergy(std::ofstream& out, unsigned int indent) {
    /*
        This component prints:
            - The number of SRAM reads
            - The number of SRAM writes

        Note that the number of times that each port is used is not shown. This is so because the use of those wires are
        taken into account in the CollectionBus and in the DSNetworkTop
   */

   counter_t reads = this->sdmemoryStats.n_SRAM_weight_reads + this->sdmemoryStats.n_SRAM_input_reads + this->sdmemoryStats.n_SRAM_psum_reads;
   counter_t writes = this->sdmemoryStats.n_SRAM_psum_writes;
   out << ind(indent) << "GLOBALBUFFER READ=" << reads; //Same line
   out << ind(indent) << " WRITE=" << writes << std::endl;
        
}

