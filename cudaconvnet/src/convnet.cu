/*
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * ---------------------------------------------------------------------------
 * Copyright 2014 Nervana Systems Inc.  All rights reserved.
 *
 * * Added timing diagnostics
 * ---------------------------------------------------------------------------
 */

#include <vector>
#include <iostream> 
#include <string>
#include <set>
#include <map>

#include "../../nvmatrix/include/nvmatrix.cuh"
#include "../../nvmatrix/include/nvmatrix_operators.cuh"
#include "../../util/include/matrix.h"
#include "../include/convnet.cuh"
#include "../include/util.cuh"

using namespace std;

/* 
 * =======================
 * ConvNet
 * =======================
 */
ConvNet::ConvNet(PyObject* layerParams, intv& deviceIDs,
                 int minibatchSize, bool conserveMem) : Thread(true) {
    _deviceIDs = deviceIDs;
    _data = NULL;
    _bufferData = NULL;
    _bufferMinibatchIdx = -1;
    _bufferPassIdx = -1;
    _trainingProgress = 0;
    _totalPassesDone = 0;
    _conserveMem = conserveMem;
    _sync = new ThreadSynchronizer(deviceIDs.size() + 1);
    PyObjectV* layerList = pyDictGetValues(layerParams);
    std::sort(layerList->begin(), layerList->end(), LayerIDComparator());

    
    _dataCopyPD = new PipeDispenserBlocking(DIVUP(_deviceIDs.size(),2)); // hard-coded for now

    initDataLayers(layerList);
    initGPUThreads(layerList);
    connectReplicas();              // Connect replicas to one another
    connectChildren(layerParams);   // Connect forward/backward links in graph
    _numFwdTerminal = 0;
    // Execute post-initialization stuff
    for (NameReplicaLayerMap::iterator it = _layerMap.begin(); it != _layerMap.end(); ++it) {
        for (int r = 0; r < it->second.size(); r++) {
            _numFwdTerminal += it->second[r]->getNext().size() == 0;
            if (it->second[r]->getNext().size() == 0) {
                printf("Fwd terminal: %s\n", it->second[r]->getName().c_str());
            }
            it->second[r]->postInit();
        }
    }

    // Find and count the terminal nodes in the backward pass
    for (int p = 0; p < getNumPasses(); p++) {
        set<Layer*> visited;
        _numBwdTerminal[p] = 0;
        for (int t = 0; t < _convNetThreads.size(); t++) {
            vector<CostLayer*>& cl = _convNetThreads[t]->getCostLayers();
            for (int c = 0; c < cl.size(); c++) {
                findBwdTerminal(*cl[c], visited, _numBwdTerminal[p], p);
            }
        }
    }

    _dp = new DataProvider(minibatchSize);
//    Py_DECREF(layerList);
    delete layerList;
}

ConvNet::~ConvNet() {
    for (vector<ConvNetThread*>::const_iterator it = _convNetThreads.begin(); it != _convNetThreads.end(); ++it) {
        (*it)->getMessageQueue().enqueue(new Message(EXIT_CONVNET));
        (*it)->join();
        delete *it;
    }
    for (DataLayerVector::const_iterator it = _dataLayers.begin(); it != _dataLayers.end(); ++it) {
        delete *it;
    }
    for (intv::const_iterator it = _deviceIDs.begin(); it != _deviceIDs.end(); ++it) {
        DEVICE_MEMORY_MANAGER::destroyInstance(*it);
    }
    HOST_MEMORY_MANAGER::destroyInstance();
    delete _sync;
    delete _dataCopyPD;
    delete _dp;
}

void ConvNet::stop() {
    getWorkerQueue().enqueue(new ExitWorker(*this));
    join();
}

PipeDispenser& ConvNet::getDataCopyPD() {
    return *_dataCopyPD;
}

void ConvNet::initDataLayers(PyObjectV* layerList) {
    for (int i = 0; i < layerList->size(); i++) {
        PyObject* paramsDict = layerList->at(i);
        std::string layerType = pyDictGetString(paramsDict, "type");

        if (layerType == "data") {
            int numReplicas = pyDictGetInt(paramsDict, "numReplicas");
            for (int r = 0; r < numReplicas; ++r) {
                DataLayer* dataLayer = new DataLayer(this, paramsDict, r);
                _dataLayers.push_back(dataLayer);
                _layerMap[dataLayer->getName()][r] = dataLayer;
            }
        }
    }
}

void ConvNet::initGPUThreads(PyObjectV* layerList) {
    // Initialize GPU worker threads
    for (int i = 0; i < _deviceIDs.size(); ++i) {
        ConvNetThread* cng = new ConvNetThread(layerList, _deviceIDs[i], i, this);
        _convNetThreads.push_back(cng);
        for (NameLayerMap::iterator it = cng->getLayerMap().begin(); it != cng->getLayerMap().end(); ++it) {
            const std::string& name = it->first;
            Layer* layer = it->second;
            _layerMap[name][layer->getReplicaID()] = layer;
        }
    }
}

void ConvNet::connectReplicas() {
    _numReplicasMax = 0;
    _numReplicasMin = 1 << 16;
    for (NameReplicaLayerMap::iterator it = _layerMap.begin(); it != _layerMap.end(); ++it) {
        _numReplicasMax = max(_numReplicasMax, int(it->second.size()));
        _numReplicasMin = min(_numReplicasMin, int(it->second.size()));
        for (map<int,Layer*>::iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2) {
            Layer& l1 = *it2->second;
            for (map<int,Layer*>::iterator it3 = it->second.begin(); it3 != it->second.end(); ++it3) {
                Layer& l2 = *it3->second;
                l1.addReplica(l2);
            }
        }
    }
}

void ConvNet::connectChildren(PyObject* layerParams) {
    for (NameReplicaLayerMap::iterator it = _layerMap.begin(); it != _layerMap.end(); ++it) {
        PyObject* paramsDict = PyDict_GetItemString(layerParams, it->first.c_str());
        PyObject* inputList = PyDict_GetItemString(paramsDict, "inputs");
        if (inputList != NULL) {
            // Iterate over "replicas" of this layer
            int numReplicas = _layerMap[it->first].size();
            for (int i = 0; i < PyList_GET_SIZE(inputList); i++) {
                std::string inputName = PyString_AsString(PyList_GetItem(inputList, i));
                int numReplicasPrev = _layerMap[inputName].size();
                // How many replicas from the previous layer must this layer be connected to?
                int numInputReplicas = numReplicasPrev / numReplicas;
                for (int r = 0; r < numReplicas; r++) {
                    for (int rp = r, ridx = 0; ridx < numInputReplicas; rp += numReplicas, ridx++) {
                        it->second[r]->addPrev(*_layerMap[inputName][rp], ridx);
                        _layerMap[inputName][rp]->addNext(*it->second[r]);
                    }
                }
            }
        }
    }
}

void ConvNet::findBwdTerminal(Layer& l, set<Layer*>& visited, int& terminal, int passIdx) {
    if (visited.count(&l) == 0) {
        visited.insert(&l);
        if (l.isGradConsumer()) {
            bool hasPrevConsumer = false;
            if (l.getPrev().size() > 0) {
                for (int i = 0; i < l.getPrev()[0].size(); i++) {
                    // Looking only at 0th replica is fine to see if you have
                    // grad consumers below you.
                    hasPrevConsumer |= l.getPrev()[0][i]->isGradConsumer();
                }
            }
            if (!hasPrevConsumer || !l.isGradProducer() || (passIdx + 1 < l.getNumReplicasPrev() && l.getNumReplicasPrev() > l.getNumReplicas())) {
                terminal++;
                l.setBwdTerminal(passIdx);
                printf("found bwd terminal %s[%d] in passIdx=%d\n", l.getName().c_str(), l.getReplicaID(), passIdx);
            } else if (l.isGradProducer()) {
                for (int r = 0; r < l.getPrev().size(); r++) {
                    for (int i = 0; i < l.getPrev()[r].size(); i++) {
                        findBwdTerminal(*l.getPrev()[r][i], visited, terminal, passIdx);
                    }
                }
            }
        }
    }
}

void* ConvNet::run() {
    for (vector<ConvNetThread*>::const_iterator it = _convNetThreads.begin(); it != _convNetThreads.end(); ++it) {
        (*it)->start();
    }
    // The manager thread defaults to using the GPU of the first worker.
    // Put more logic here if this is inappropriate.
    NVMatrix::setDeviceID(_convNetThreads[0]->getDeviceID());
    copyToGPU();
    bool exit = false;
    while (!exit) {
        Worker* worker = _workerQueue.dequeue();
        exit = worker->run();
        delete worker;
    }

    return NULL;
}

Queue<Worker*>& ConvNet::getWorkerQueue() {
    return _workerQueue;
}

Queue<WorkResult*>& ConvNet::getResultQueue() {
    return _resultQueue;
}

DataProvider& ConvNet::getDataProvider() {
    return *_dp;
}

Layer& ConvNet::getLayer(std::string& name, int replicaID) {
    return *_layerMap[name][replicaID];
}

void ConvNet::sendMessage(MESSAGES msg, bool sync) {
    sendMessage(new Message(msg), sync);
}

void ConvNet::sendMessage(Message* msg, bool sync) {
    for (int i = 0; i < _convNetThreads.size(); i++) {
        _convNetThreads[i]->getMessageQueue().enqueue(msg->clone());
    }

    delete msg;

    if (sync) {
        syncWithChildren();
    }
}

void ConvNet::copyToCPU() {
    sendMessage(COPY_TO_CPU, true);
}

void ConvNet::copyToGPU() {
    sendMessage(COPY_TO_GPU, false);
}

void ConvNet::updateWeights(int passIdx) {
    sendMessage(UPDATE_WEIGHTS, true);
    sendMessage(CONSTRAIN_WEIGHTS, true);
}

void ConvNet::reset(int passIdx) {
    sendMessage((passIdx % getNumPasses()) == 0 ? RESET : RESET_PASS_IDX, false);
}

void ConvNet::reset() {
    reset(0);
}

// Fprop given data
void ConvNet::fprop(CPUData& data, int passIdx, PASS_TYPE passType) {
    reset(passIdx);
    // This is necessary because setData below could delete data. If there's
    // an outstanding copy request, this'll cause a segfault.
    for (int i = 0; i < _dataLayers.size(); i++) {
        _dataLayers[i]->waitForCopyFinish();
    }

    setData(data, passIdx);
    for (int i = 0; i < _dataLayers.size(); i++) {
        _dataLayers[i]->fprop(passType, passIdx, false);
    }
    waitForTerminals(_numFwdTerminal, FPROP_TERMINAL);
}

// Fprop given minibatch idx
void ConvNet::fprop(int miniIdx, int passIdx, PASS_TYPE passType) {
    reset(passIdx);

    bool fromBuffer = miniIdx == _bufferMinibatchIdx && passIdx == _bufferPassIdx;
    if (!fromBuffer) {
        // This is necessary because setData below could delete data. If there's
        // an outstanding copy request, this'll cause a segfault.
        for (int i = 0; i < _dataLayers.size(); i++) {
            _dataLayers[i]->waitForCopyFinish();
        }

        setData(_dp->getMinibatch(miniIdx), passIdx);

    } else {
        setDataFromBuffer();
    }
    for (int i = 0; i < _dataLayers.size(); i++) {
        _dataLayers[i]->fprop(passType, passIdx, fromBuffer);
    }

    if (passIdx == getNumPasses() - 1) {
        // Do double-buffering from next minibatch from the DataProvider
        setBuffer(miniIdx == _dp->getNumMinibatches() - 1 ? NULL : &_dp->getMinibatch(miniIdx + 1), miniIdx + 1, 0);
    } else {
        // Do double-buffering from next microbatch within current minibatch
        setBuffer(_data, miniIdx, passIdx + 1);
    }

    waitForTerminals(_numFwdTerminal, FPROP_TERMINAL);
}

void ConvNet::setDataFromBuffer() {
    if (_bufferData != _data) {
        delete _data;
    }
    _data = _bufferData;
    _bufferData = NULL;
    _bufferMinibatchIdx = -1;
    _bufferPassIdx = -1;
}

void ConvNet::setData(CPUData& data, int passIdx) {
    bool same = _data == _bufferData;
    if (&data != _data) {
        delete _data;
    }
    if (&data != _bufferData && !same) {
        delete _bufferData;
        _bufferData = NULL;
        _bufferMinibatchIdx = -1;
        _bufferPassIdx = -1;
    }
    _data = &data;
    for (int i = 0; i < _dataLayers.size(); i++) {
        _dataLayers[i]->copyData(*_data, false, passIdx);
    }
}

void ConvNet::setBuffer(CPUData* bufferData, int bufferMinibatchIdx, int bufferPassIdx) {
    _bufferData = bufferData;
    _bufferMinibatchIdx = bufferMinibatchIdx;
    _bufferPassIdx = bufferPassIdx;
    if (bufferData != NULL) {
        for (int i = 0; i < _dataLayers.size(); i++) {
            _dataLayers[i]->copyData(*_bufferData, true, bufferPassIdx);
        }
    }
}

CPUData& ConvNet::getData() {
    assert(_data != NULL);
    return *_data;
}

void ConvNet::bprop(int passIdx, PASS_TYPE passType) {
    _totalPassesDone++;
    sendMessage(new BpropStartMessage(passType, passIdx), false);
    waitForTerminals(_numBwdTerminal[passIdx], BPROP_TERMINAL);
    reset(passIdx + 1);
}

void ConvNet::waitForTerminals(int numMsgs, MESSAGES msgType) {
    for (int rcvd = 0; rcvd < numMsgs; rcvd++) {
        Message* m = _msgQueue.dequeue();
        assert(m->getType() == msgType);
        delete m;
    }
}

// Same as getCost() but adds results to given cost and returns it
Cost& ConvNet::getCost(Cost& cost) {
    Cost &tmp = getCost();
    cost += tmp;
    delete &tmp;
    return cost;
}

Cost& ConvNet::getCost() {
    Cost& cost = *new Cost();
    for (int t = 0; t < _convNetThreads.size(); t++) {
        Cost& tcost = _convNetThreads[t]->getCost();
        cost += tcost;
        delete &tcost;
    }
    return cost;
}

double ConvNet::getCostValue() {
    Cost& cost = getCost();
    double val = cost.getValue();
    delete &cost;
    return val;
}

Queue<Message*>& ConvNet::getMessageQueue() {
    return _msgQueue;
}

intv& ConvNet::getDeviceIDs() {
    return _deviceIDs;
}

ThreadSynchronizer& ConvNet::getSync() {
    return *_sync;
}

void ConvNet::syncWithChildren() {
    sendMessage(SYNC, false);
    _sync->sync();
}

int ConvNet::getTotalPassesDone() {
    return _totalPassesDone;
}

int ConvNet::getMinibatchSize() {
    return _dp->getMinibatchSize();
}

int ConvNet::getNumReplicasMax() {
    return _numReplicasMax;
}

int ConvNet::getNumReplicasMin() {
    return _numReplicasMin;
}

int ConvNet::getNumPasses() {
    return _numReplicasMax / _numReplicasMin;
}

void ConvNet::setTrainingProgress(double progress) {
    _trainingProgress = progress;
}

double ConvNet::getTrainingProgress() const {
    return _trainingProgress;
}

bool ConvNet::isConserveMemory() {
    return _conserveMem;
}

/*
 * Gradient checking stuff
 */
void ConvNet::checkGradients() {
    _numFailures = 0;
    _numTests = 0;
    _baseErr = 0;
    for (int p = 0; p < getNumPasses(); ++p) {
        fprop(0, p, PASS_GC);
        _baseErr += getCostValue();
        bprop(p, PASS_GC);
    }
    // We call grad check only on the first replica,
    // but because weights are aware of their fellow replicas,
    // we can simultaneously perturb the weights of all
    // replicas.
    for (NameReplicaLayerMap::iterator it = _layerMap.begin(); it != _layerMap.end(); ++it) {
        map<int, Layer*>& layers = it->second;
        if (layers[0]->getDeviceID() >= 0 /*&& (layers[0]->getName() == "fc10")*/) { // If layer on GPU (data layers aren't)
            layers[0]->checkGradient();
        }
    }

    cout << "------------------------" << endl;
    if (_numFailures > 0) {
        cout << _numFailures << "/" << _numTests << " TESTS FAILED" << endl;
    } else {
        cout << "ALL " << _numTests << " TESTS PASSED" << endl;
    }
}

// Copies to all replicas
void ConvNet::checkGradient_copyWeightsToGPU(Matrix& weightsCPU, Weights& weights) {
    int d = NVMatrix::getDeviceID();
    for (map<int, Weights*>::const_iterator it = weights.getReplicas().begin(); it != weights.getReplicas().end(); ++it) {
        NVMatrix::setDeviceID(it->second->getDeviceID());
        it->second->getW().copyFromHost(weightsCPU);
    }
    NVMatrix::setDeviceID(d);
}

/*
 * name: weight matrix name
 * eps: finite difference step
 */
bool ConvNet::checkGradient(const std::string& name, float eps, Weights& weights) {
    Matrix numGrad(weights.getNumRows(), weights.getNumCols());
    Matrix diff(numGrad);
    numGrad.apply(Matrix::ZERO);
    Matrix weightsCPU;

    weights.getW().copyToHost(weightsCPU, true);

    for(int i = 0; i < weights.getNumRows(); i++) {
        for (int j = 0; j < weights.getNumCols(); j++) {
            float v = weightsCPU(i,j);
            weightsCPU(i,j) += eps;

            checkGradient_copyWeightsToGPU(weightsCPU, weights);

            weightsCPU(i,j) = v;
            double err = 0;
            for (int p = 0; p < getNumPasses(); ++p) {
//                printf("trying fprop %d\n", p);
                fprop(0, p, PASS_GC);
//                printf("    success\n");
                err += getCostValue();
            }
            numGrad(i,j) = (err - _baseErr) / (_data->getNumCases() * eps);
            if (isnan((double)numGrad(i,j)) || isinf((double)numGrad(i,j))) {
                cout << "Numerical computation produced nan or inf when checking '" << name << "': " << numGrad(i,j) << endl;
                cout << "Consider reducing the sizes of the weights or finite difference steps." << endl;
                cout << "Exiting." << endl;
                exit(1);
            }
            checkGradient_copyWeightsToGPU(weightsCPU, weights);
        }
    }
    Matrix gradCPU;
    NVMatrix::setDeviceID(weights.getDeviceID());
    map<int,NVMatrix*> mats;
    for (map<int, Weights*>::const_iterator it = weights.getReplicas().begin(); it != weights.getReplicas().end(); ++it) {
        mats[it->first] = &it->second->getGrad();
    }
    weights.getReducer().reduce(mats, 1, false);

    weights.getGrad().copyToHost(gradCPU, true);
    gradCPU.scale(-1.0 / _data->getNumCases());
    float analNorm = gradCPU.norm();
    float numNorm = numGrad.norm();
    numGrad.subtract(gradCPU, diff);
    float relErr = diff.norm() / analNorm;
    bool fail = relErr >= GC_REL_ERR_THRESH;
    if (fail || !GC_SUPPRESS_PASSES) {
        cout << "========================" << endl;
        printf("(%s) %s GRADIENT CHECK\n", fail ? "****FAIL****" : "PASS", name.c_str());
        cout << "========================" << endl;
        cout << "Analytic:" << endl;
        gradCPU.print(0, 6, 0, 4);
        cout << "Numeric:" << endl;
        numGrad.print(0, 6, 0, 4);
        printf("Analytic norm: %e\n", analNorm);
        printf("Numeric norm:  %e\n", numNorm);
        printf("Relative error: %e\n", relErr);
    }
    _numTests++;
    _numFailures += fail;
    return fail;
}

/* 
 * =======================================================================================================
 * ConvNetThread
 * =======================================================================================================
 */
ConvNetThread::ConvNetThread(PyObjectV* layerList, int deviceID, int deviceIdx, ConvNet* convNet)
    : Thread(true, getDeviceCPUs(deviceID)), _deviceID(deviceID), _convNet(convNet) {
    try {
        int numLayers = layerList->size();

        for (int i = 0; i < numLayers; i++) {
            PyObject* paramsDict = layerList->at(i);
            std::string layerType = pyDictGetString(paramsDict, "type");
            if (layerType != "data") {
                intv& gpus = *pyDictGetIntV(paramsDict, "gpu");
                int rid = indexOf(gpus, deviceIdx);
                if (rid >= 0) {
                    initLayer(paramsDict, rid);
                }
                delete &gpus;
            }
        }
    } catch (std::string& s) {
        cout << "Error creating ConvNet: " << s << endl;
        exit(1);
    }
}

ConvNetThread::~ConvNetThread() {
    NVMatrix::setDeviceID(_deviceID);
    NVMatrix::destroyCublas();
    NVMatrix::destroyRandom();
    for (NameLayerMap::const_iterator it = _nameLayerMap.begin(); it != _nameLayerMap.end(); ++it) {
        delete it->second;
    }
    _nameLayerMap.clear();
}

void ConvNetThread::startTimer() {
    NVMatrix::syncStream();
    _timer.start();
}

double ConvNetThread::stopTimer() {
    NVMatrix::syncStream();
    return _timer.stop();
}

void ConvNetThread::initLayer(PyObject* paramsDict, int replicaID) {
    std::string type = pyDictGetString(paramsDict, "type");
    std::string name = pyDictGetString(paramsDict, "name");
    if (type == "fc") {
        _nameLayerMap[name] = new FCLayer(this, paramsDict, replicaID, false);
    } else if (type == "sfc") {
        _nameLayerMap[name] = new SplitFCLayer(this, paramsDict, replicaID, false);
    } else if (type == "conv") {
        _nameLayerMap[name] = new ConvLayer(this, paramsDict, replicaID);
    } else if (type == "local") {
        _nameLayerMap[name] = new LocalUnsharedLayer(this, paramsDict, replicaID);
    } else if (type == "pool") {
        _nameLayerMap[name] = &PoolLayer::make(this, paramsDict, replicaID);
    } else if (type == "cmpool") {
        _nameLayerMap[name] = &CrossMapPoolLayer::make(this, paramsDict, replicaID);
    } else if (type == "rnorm") {
        _nameLayerMap[name] = new ResponseNormLayer(this, paramsDict, replicaID);
    } else if (type == "cmrnorm") {
        _nameLayerMap[name] = new CrossMapResponseNormLayer(this, paramsDict, replicaID);
    } else if (type == "cnorm") {
        _nameLayerMap[name] = new ContrastNormLayer(this, paramsDict, replicaID);
    } else if (type == "softmax") {
        _nameLayerMap[name] = new SoftmaxLayer(this, paramsDict, replicaID);
    } else if (type == "eltsum") {
        _nameLayerMap[name] = new EltwiseSumLayer(this, paramsDict, replicaID);
    } else if (type == "eltmax") {
        _nameLayerMap[name] = new EltwiseMaxLayer(this, paramsDict, replicaID);
    } else if (type == "neuron") {
        _nameLayerMap[name] = new NeuronLayer(this, paramsDict, replicaID);
    } else if (type == "nailbed") {
        _nameLayerMap[name] = new NailbedLayer(this, paramsDict, replicaID);
    } else if (type == "blur") {
        _nameLayerMap[name] = new GaussianBlurLayer(this, paramsDict, replicaID);
    } else if (type == "href") {
        _nameLayerMap[name] = new HorizontalReflectionLayer(this, paramsDict, replicaID);
    } else if (type == "resize") {
        _nameLayerMap[name] = new ResizeLayer(this, paramsDict, replicaID);
    } else if (type == "rgb2yuv") {
        _nameLayerMap[name] = new RGBToYUVLayer(this, paramsDict, replicaID);
    } else if (type == "rgb2lab") {
        _nameLayerMap[name] = new RGBToLABLayer(this, paramsDict, replicaID);
    } else if (type == "rscale") {
        _nameLayerMap[name] = new RandomScaleLayer(this, paramsDict, replicaID);
    } else if (type == "crop") {
        _nameLayerMap[name] = new CropLayer(this, paramsDict, replicaID);
    } else if (type == "concat") {
        _nameLayerMap[name] = new ConcatenationLayer(this, paramsDict, replicaID);
    } else if (type == "pass") {
        _nameLayerMap[name] = new PassThroughLayer(this, paramsDict, replicaID);
    } else if (type == "dropout") {
        _nameLayerMap[name] = new DropoutLayer(this, paramsDict, replicaID);
    } else if (type == "dropout2") {
        _nameLayerMap[name] = new Dropout2Layer(this, paramsDict, replicaID);
    } else if (strncmp(type.c_str(), "cost.", 5) == 0) {
        CostLayer *c = &CostLayer::make(this, paramsDict, type, replicaID);
        _nameLayerMap[name] = c;
        _costs.push_back(c);
    } else {
        throw std::string("Unknown layer type ") + type;
    }
}

/*
 * This executes in a new CPU thread so it's OK to initialize CUDA stuff here. 
 */
void ConvNetThread::initCuda() { 
    NVMatrix::setDeviceID(_deviceID);
    checkCudaErrors(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
    for (int i = 0; i < _convNet->getDeviceIDs().size(); i++) {
        int d = _convNet->getDeviceIDs()[i];
        if (d != _deviceID) {
            if (NVMatrix::canAccessPeer(_deviceID, d)) {
                printf("Enabling peer access GPU %d --> GPU %d\n", NVMatrix::getDeviceID(), d);
                checkCudaErrors(cudaDeviceEnablePeerAccess(d, 0));
            } else {
                printf("No peer access GPU %d -->  GPU %d\n", _deviceID, d);
            }
        }
    }
//    NVMatrix::syncStream();
    NVMatrix::initCublas();
    NVMatrix::initRandom(/*7*/);
    srand(time(0));
}

// (u) helper function for timing stuff
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
//

void* ConvNetThread::run() {
    initCuda();
    bool exit = false;
    // (u) setting up some timing stuff
    bool timingdump = false;
    float time;
    float time0=0;
    float time1=0;
    float time2=0;
    float time3=0;
    float time4=0;
    float time5=0;
    float time6=0;
    float time7=0;
    float time8=0;
    float time9=0;
    cudaEvent_t start, stop;
    //
    while (!exit) {
        Message* m = _msgQueue.dequeue();
        if (timingdump) { printf("."); }
        if (m->getType() == FPROP_READY) {
            if (timingdump) { printf("0");
            HANDLE_ERROR( cudaEventCreate(&start) ); HANDLE_ERROR( cudaEventCreate(&stop) ); HANDLE_ERROR( cudaEventRecord(start, 0) );}
            FpropMessage* msg = static_cast<FpropMessage*>(m);
            msg->getToLayer().fprop(msg->getPassType(), msg->getPassIdx());
            if (timingdump) { HANDLE_ERROR( cudaEventRecord(stop, 0) ); HANDLE_ERROR( cudaEventSynchronize(stop) ); HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) ); time0+=time;}
        } else if (m->getType() == BPROP_READY) {
            if (timingdump) { printf("1");
            HANDLE_ERROR( cudaEventCreate(&start) ); HANDLE_ERROR( cudaEventCreate(&stop) ); HANDLE_ERROR( cudaEventRecord(start, 0) );}
            BpropMessage* msg = static_cast<BpropMessage*>(m);
            msg->getToLayer().incRcvdBInputMsgs();
            msg->getToLayer().bprop(msg->getPassType(), msg->getPassIdx());
            if (timingdump) { HANDLE_ERROR( cudaEventRecord(stop, 0) ); HANDLE_ERROR( cudaEventSynchronize(stop) ); HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) ); time1+=time;} 
        } else if (m->getType() == BPROP_START) {
            if (timingdump) { printf("2");
            HANDLE_ERROR( cudaEventCreate(&start) ); HANDLE_ERROR( cudaEventCreate(&stop) ); HANDLE_ERROR( cudaEventRecord(start, 0) );}
            BpropStartMessage* msg = static_cast<BpropStartMessage*>(m);
            for (int i = 0; i < _costs.size(); i++) {
                dynamic_cast<Layer*>(_costs[i])->bprop(msg->getPassType(), msg->getPassIdx());
            }
            if (timingdump) { HANDLE_ERROR( cudaEventRecord(stop, 0) ); HANDLE_ERROR( cudaEventSynchronize(stop) ); HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) ); time2+=time;}
        } else if (m->getType() == SYNC) {
            if (timingdump) { printf("3");
            HANDLE_ERROR( cudaEventCreate(&start) ); HANDLE_ERROR( cudaEventCreate(&stop) ); HANDLE_ERROR( cudaEventRecord(start, 0) );}
            NVMatrix::syncStream();
            _convNet->getSync().sync();
            if (timingdump) { HANDLE_ERROR( cudaEventRecord(stop, 0) ); HANDLE_ERROR( cudaEventSynchronize(stop) ); HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) ); time3+=time;} 
        } else if (m->getType() == COPY_TO_CPU) {
            if (timingdump) { printf("4");
            HANDLE_ERROR( cudaEventCreate(&start) ); HANDLE_ERROR( cudaEventCreate(&stop) ); HANDLE_ERROR( cudaEventRecord(start, 0) );}
            for (NameLayerMap::iterator it = _nameLayerMap.begin(); it != _nameLayerMap.end(); ++it) {
                it->second->copyToCPU();
            }
            if (timingdump) { HANDLE_ERROR( cudaEventRecord(stop, 0) ); HANDLE_ERROR( cudaEventSynchronize(stop) ); HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) ); time4+=time;} 
        } else if (m->getType() == COPY_TO_GPU) {
            if (timingdump) { printf("5");
            HANDLE_ERROR( cudaEventCreate(&start) ); HANDLE_ERROR( cudaEventCreate(&stop) ); HANDLE_ERROR( cudaEventRecord(start, 0) );}
            for (NameLayerMap::iterator it = _nameLayerMap.begin(); it != _nameLayerMap.end(); ++it) {
                it->second->copyToGPU();
            }
            if (timingdump) { HANDLE_ERROR( cudaEventRecord(stop, 0) ); HANDLE_ERROR( cudaEventSynchronize(stop) ); HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) ); time5+=time;} 
        } else if (m->getType() == RESET) {
            if (timingdump) { printf("6");
            HANDLE_ERROR( cudaEventCreate(&start) ); HANDLE_ERROR( cudaEventCreate(&stop) ); HANDLE_ERROR( cudaEventRecord(start, 0) );}
            for (NameLayerMap::iterator it = _nameLayerMap.begin(); it != _nameLayerMap.end(); ++it) {
                it->second->reset();
            }
            if (timingdump) { HANDLE_ERROR( cudaEventRecord(stop, 0) ); HANDLE_ERROR( cudaEventSynchronize(stop) ); HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) ); time6+=time;}  
        } else if (m->getType() == RESET_PASS_IDX) {
            if (timingdump) { printf("7");
            HANDLE_ERROR( cudaEventCreate(&start) ); HANDLE_ERROR( cudaEventCreate(&stop) ); HANDLE_ERROR( cudaEventRecord(start, 0) );}
            for (NameLayerMap::iterator it = _nameLayerMap.begin(); it != _nameLayerMap.end(); ++it) {
                it->second->resetPassIdx();
            }
            if (timingdump) { HANDLE_ERROR( cudaEventRecord(stop, 0) ); HANDLE_ERROR( cudaEventSynchronize(stop) ); HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) ); time7+=time;}
        } else if (m->getType() == UPDATE_WEIGHTS) {
            if (timingdump) { printf("8");
            HANDLE_ERROR( cudaEventCreate(&start) ); HANDLE_ERROR( cudaEventCreate(&stop) ); HANDLE_ERROR( cudaEventRecord(start, 0) );}
            for (NameLayerMap::iterator it = _nameLayerMap.begin(); it != _nameLayerMap.end(); ++it) {
                it->second->updateWeights();
            }
            if (timingdump) { HANDLE_ERROR( cudaEventRecord(stop, 0) ); HANDLE_ERROR( cudaEventSynchronize(stop) ); HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) ); time8+=time;}
        } else if (m->getType() == CONSTRAIN_WEIGHTS) {
            if (timingdump) { printf("9"); 
            HANDLE_ERROR( cudaEventCreate(&start) ); HANDLE_ERROR( cudaEventCreate(&stop) ); HANDLE_ERROR( cudaEventRecord(start, 0) ); }
            for (NameLayerMap::iterator it = _nameLayerMap.begin(); it != _nameLayerMap.end(); ++it) {
                it->second->constrainWeights();
            }
            if (timingdump) { HANDLE_ERROR( cudaEventRecord(stop, 0) ); HANDLE_ERROR( cudaEventSynchronize(stop) ); HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) ); time9+=time; }
            // dump all the text here! 
            if (timingdump) { 
                printf("\n------------ timing -------------- \n");
                printf("Time0 to fprop ready:  %3.1f ms \n", time0);
                printf("Time1 to bprop ready:  %3.1f ms \n", time1);
                printf("Time2 to bprop start:  %3.1f ms \n", time2);
                printf("Time3 to sync:  %3.1f ms \n", time3);
                printf("Time4 to copy to cpu:  %3.1f ms \n", time4);
                printf("Time5 to copy to gpu:  %3.1f ms \n", time5);
                printf("Time6 to reset:  %3.1f ms \n", time6);
                printf("Time7 to reset index:  %3.1f ms \n", time7);
                printf("Time8 to update weights:  %3.1f ms \n", time8);
                printf("Time9 to constrain:  %3.1f ms \n", time9);
                time0 = time1 = time2 = time3 = time4 = time5 = time6 = time7 = time8 = time9 = 0;
            }
            // here we get output at every minibatch, great!         
        } else if (m->getType() == EXIT_CONVNET) {
            exit = true;
        }
        delete m;
    }
    return NULL;
}

Cost& ConvNetThread::getCost() {
    // In a single ConvNetThread, all costs are guaranteed to be different
    // (i.e. not replicas of one another)
    return *new Cost(_costs);
}

Layer& ConvNetThread::getLayer(std::string& name) {
    return *_nameLayerMap[name];
}

int ConvNetThread::getDeviceID() {
    return _deviceID;
}

Queue<Message*>& ConvNetThread::getMessageQueue() {
    return _msgQueue;
}

vector<CostLayer*>& ConvNetThread::getCostLayers() {
    return _costs;
}

NameLayerMap& ConvNetThread::getLayerMap() {
    return _nameLayerMap;
}

ConvNet& ConvNetThread::getConvNet() {
    return *_convNet;
}

