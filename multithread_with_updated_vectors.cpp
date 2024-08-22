#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <ctime>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <matplotlibcpp.h>

namespace plt = matplotlibcpp;

// 全域變數
bool useGPU;
std::string inputFile;
int numChannels;
torch::Device device = torch::kCPU;
int picCount = 0;

// 佇列及同步原語
std::queue<std::vector<std::vector<float>>> dataQueue;
std::mutex queueMutex;
std::condition_variable dataCondition;
bool doneReading = false;

std::vector<std::vector<float>> read_csv(int numCol, std::ifstream& file);
void FFTfunction(std::vector<std::vector<float>> data, int numChannels);
torch::Device getDevice(bool b);
void moveDataToGPU(std::vector<torch::Tensor>& data_tensors, torch::Device device);
void moveDataToCPU(std::vector<torch::Tensor>& data_tensors);
void extractPositiveFreqAndMag(const std::vector<float>& freq, const std::vector<float>& magnitude, std::vector<float>& pos_freq, std::vector<float>& pos_mag);
int find_peak_index(const std::vector<float>& magnitudes);
void plotFFT(const std::vector<float>& freq, const std::vector<float>& magnitude, int peak_index, const std::string& title, const std::string& color, int i);

void readerThreadFunc(int numCol, std::ifstream& file) {
    time_t totalRead = 0;
    while (true) {
        auto start = clock();
        auto data = read_csv(numCol, file);
        totalRead += clock() - start;
        if (data[0].empty()) {
            break;
        }
        std::unique_lock<std::mutex> lock(queueMutex);
        //auto push = clock();

        dataQueue.push(data);
        //std::cout << "move to queue:  " << clock() - push << '\n';

        lock.unlock();
        dataCondition.notify_one();

    }
    doneReading = true;
    dataCondition.notify_all();
    std::cout << "read_file: " << totalRead * 12.5 << '\n';

}

void processorThreadFunc(int numChannels) {
    //auto read_start = clock();
    while (true) {
        std::vector<std::vector<float>> data;

        std::unique_lock<std::mutex> lock(queueMutex);
        dataCondition.wait(lock, [] { return !dataQueue.empty() || doneReading; });

        if (dataQueue.empty() && doneReading) {
            break;
        }

        data = std::move(dataQueue.front());
        dataQueue.pop();

        //std::cout << "running fft " << std::endl;
        FFTfunction(data, numChannels);
    }
    //std::cout << "read time: " << clock() - read_start << '\n';
}

int main() {
    std::cout << "Enter the number of channels (excluding time): ";
    std::cin >> numChannels;

    std::cout << "Please input the inputFile name (including csv): ";
    std::cin >> inputFile;

    std::cout << "Please choose device (0: cpu, 1: gpu):  ";
    std::cin >> useGPU;

    device = getDevice(useGPU);
    std::ifstream file(inputFile);

    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + inputFile);
    }

    // Skip the header lines
    while (std::getline(file, inputFile)) {
        if (inputFile.find("Label") != std::string::npos) {
            break;
        }
    }
    std::getline(file, inputFile); // Skip the label line

    auto start = clock();

    // 啟動多執行緒
    std::thread readerThread(readerThreadFunc, numChannels + 1, std::ref(file));
    std::thread processorThread(processorThreadFunc, numChannels);

    readerThread.join();
    processorThread.join();

    file.close();
    std::cout << "time: " << (clock() - start) * 12.5;

    system("pause");
    return 0;
}

std::vector<std::vector<float>> read_csv(int numCol, std::ifstream& file) {
    std::vector<std::vector<float>> data(numChannels+1);
    
    std::string line, cell;
    int count = 0;

    while (std::getline(file, line) && count < 4000) {
        std::stringstream lineStream(line);
        for (int i = 0; i < numChannels + 1 && std::getline(lineStream, cell, ','); i++) {
            try {
                data[i].push_back(std::stof(cell));
            }
            catch (const std::invalid_argument& e) {
                std::cerr << "Invalid data: " << cell << " in line: " << line << std::endl;
                throw;
            }
        }
        count++;
    }

    return data;
}

void FFTfunction(std::vector<std::vector<float>> data, int numChannels) {
    //std::cout << picCount << '\n';
    picCount++;

    std::vector<torch::Tensor> data_tensors;
    data_tensors.push_back(torch::from_blob(data[0].data(), {static_cast<int64_t>(data[0].size())}, torch::kFloat32));
     
    for (int i = 0; i < numChannels; i++) {
        //std::cout << i << '\n';
        data_tensors.push_back(torch::from_blob(data[i + 1].data(), { static_cast<int64_t>(data[i + 1].size()) }, torch::kFloat32));
    }

    if (useGPU) {
        moveDataToGPU(data_tensors, device);
    }
    time_t move_to_gpu = clock();

    for (int i = 1; i <= numChannels; ++i) {
        float deltaT = data_tensors[0][1].item<float>() - data_tensors[0][0].item<float>();
        if (deltaT == 0) {
            std::cerr << "Delta time is zero, invalid for frequency calculation." << std::endl;
            continue;
        }

        torch::Tensor time_tensor = torch::fft::fft(data_tensors[0].to(device));
        torch::Tensor fft_result = torch::fft::fft(data_tensors[i].to(device));
        torch::Tensor fft_freq = torch::fft::fftfreq(data_tensors[i].size(0), deltaT).to(device);
        time_t fft = clock();
        /* 
         ㄒ
        fft_result = fft_result.to(torch::kCPU);
        fft_freq = fft_freq.to(torch::kCPU);

        moveDataToCPU(data_tensors);
        time_t back_to_cpu = clock();

        torch::Tensor fft_magnitude = torch::abs(fft_result);
        std::vector<float> freq_vec(fft_freq.data_ptr<float>(), fft_freq.data_ptr<float>() + fft_freq.numel());
        std::vector<float> magnitude_vec(fft_magnitude.data_ptr<float>(), fft_magnitude.data_ptr<float>() + fft_magnitude.numel());
        time_t turn_to_vector = clock();

        std::vector<float> print_freq, print_mag;
        extractPositiveFreqAndMag(freq_vec, magnitude_vec, print_freq, print_mag);

        time_t find_peak = clock();

        int peak_index = find_peak_index(print_mag);
        plotFFT(print_freq, print_mag, peak_index, "FFT of CH" + std::to_string(i), "b", i);
        time_t show_figure = clock();

        
        std::cout << "time to move data to tensor: " << move_to_gpu - read_file << " milliseconds" << '\n';
        std::cout << "time to compute fft: " << fft - move_to_gpu << " milliseconds" << '\n' << '\n';

        std::cout << "time to move data back to cpu : " << back_to_cpu - fft << " milliseconds" << '\n';
        std::cout << "time to turn data into vectors : " << turn_to_vector - back_to_cpu << " milliseconds" << '\n';
        std::cout << "time to find peak : " << find_peak - turn_to_vector << " milliseconds" << '\n';
        std::cout << "time to plot figure: " << show_figure - find_peak << " milliseconds" << "\n\n";
        */
    }
}

torch::Device getDevice(bool b) {
    if (b && torch::cuda::is_available()) {
        std::cout << "CUDA is available! Using GPU." << std::endl;
        return torch::Device(torch::kCUDA);
    }
    else {
        std::cout << "Using CPU." << std::endl;
        return torch::Device(torch::kCPU);
    }
}

void moveDataToGPU(std::vector<torch::Tensor>& data_tensors, torch::Device device) {
    for (auto& tensor : data_tensors) {
        tensor = tensor.clone().to(device);
    }
}

void moveDataToCPU(std::vector<torch::Tensor>& data_tensors) {
    for (auto& tensor : data_tensors) {
        tensor = tensor.to(torch::kCPU);
    }
}

void extractPositiveFreqAndMag(const std::vector<float>& freq, const std::vector<float>& magnitude, std::vector<float>& pos_freq, std::vector<float>& pos_mag) {
    for (size_t i = 0; i < freq.size(); ++i) {
        if (freq[i] > 0) {
            pos_freq.push_back(freq[i]);
            pos_mag.push_back(magnitude[i]);
        }
    }
}

int find_peak_index(const std::vector<float>& magnitudes) {
    int peak_index = 0;
    for (size_t i = 1; i < magnitudes.size(); ++i) {
        if (magnitudes[i] > magnitudes[peak_index]) {
            peak_index = i;
        }
    }
    return peak_index;
}

void plotFFT(const std::vector<float>& freq, const std::vector<float>& magnitude, int peak_index, const std::string& title, const std::string& color, int i) {
    plt::clf();
    plt::plot(freq, magnitude, color + "-");
    plt::annotate("Peak (" + std::to_string(freq[peak_index]) + ", " + std::to_string(magnitude[peak_index]) + ")",
        freq[peak_index], magnitude[peak_index]);
    plt::title(title);
    plt::xlabel("Frequency (Hz)");
    plt::ylabel("Amplitude");

    std::string outputFile = title + std::to_string(picCount);
    //std::cout << "Please input the outputFile name for " << i << ": ";

    //std::cin >> outputFile;

    plt::save(outputFile);

    //plt::show();
}