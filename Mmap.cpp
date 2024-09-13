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
#include "MemoryMapped.h"

namespace plt = matplotlibcpp;


bool useGPU;
std::string inputFile;
int numChannels;
torch::Device device = torch::kCPU;
int picCount = 0;


std::queue<std::vector<std::vector<float>>> dataQueue;
std::mutex queueMutex;
std::condition_variable dataCondition;
bool doneReading = false;

std::vector<std::vector<float>> read_csv(int numcol, MemoryMapped& dataChar);
void FFTfunction(std::vector<std::vector<float>> data, int numChannels);
torch::Device getDevice(bool b);
void moveDataToGPU(std::vector<torch::Tensor>& data_tensors, torch::Device device);
void moveDataToCPU(std::vector<torch::Tensor>& data_tensors);
void extractPositiveFreqAndMag(const std::vector<float>& freq, const std::vector<float>& magnitude, std::vector<float>& pos_freq, std::vector<float>& pos_mag);
int find_peak_index(const std::vector<float>& magnitudes);
void plotFFT(const std::vector<float>& freq, const std::vector<float>& magnitude, int peak_index, const std::string& title, const std::string& color, int i);

void readerThreadFunc(int numCol, MemoryMapped& dataChar) {
    while (true)
    {
        time_t totalRead = 0;
        auto  now_size = dataChar.size();
        while (true) {
            auto start = clock();
            auto data = read_csv(numCol, dataChar);
            totalRead += clock() - start;
            if (data[0][1] == 0) {
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
        dataChar.open(inputFile);
        if (now_size == dataChar.size())
        {
            break;
        }
    }


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


int count = 0;
int main() {
    std::cout << "Enter the number of channels (excluding time): ";
    std::cin >> numChannels;

    std::cout << "Please input the inputFile name (including csv): ";
    std::cin >> inputFile;

    std::cout << "Please choose device (0: cpu, 1: gpu):  ";
    std::cin >> useGPU;

    device = getDevice(useGPU);
    MemoryMapped dataChar(inputFile);

    //std::cout << "data.size() =  " << data.size() << std::endl;
    if (dataChar.size() > 0) {
        while (count < dataChar.size()) {
            if (dataChar[count] == 'T' /*&& dataChar[count + 1] == 'I' && dataChar[count + 2] == 'M' && dataChar[count + 3] == 'E'*/) {
                break;
            }
            count++;
        }

        //std::cout << "data[count + 4]  " << data[count + 5] << std::endl;
        count = count + 4;
        while (true) {
            if (dataChar[count] == '\n') {
                break;
            }
            count++;
        }
        count++;
    }

    auto start = clock();

    std::thread readerThread(readerThreadFunc, numChannels + 1, std::ref(dataChar));
    std::thread processorThread(processorThreadFunc, numChannels);

    readerThread.join();
    processorThread.join();

    std::cout << "time: " << (clock() - start) * 5;
    //std::cout << "picCount: " << picCount;

    system("pause");
    return 0;
}

int readCount = 0;
std::vector<std::vector<float>> read_csv(int numcol, MemoryMapped& dataChar) {
    std::vector<std::vector<float>> datas(numcol,std::vector<float>(4000,0));  // 创建numcol个向量，每个用于存储一列的数据
    std::string buff = "";  // 用于存储当前正在处理的数字
    int currentCol = 0;  // 用于追踪当前列
    int lineCount = 0;  // 用于记录当前的行数
    // 追踪 MemoryMapped 中的字符位置

    while (count < dataChar.size() && lineCount < 4000) {  // 限制读取的行数
        char ch = dataChar[count];
        if (ch == ',') {
            // 遇到逗号，转换buff为float并放入当前列
            if (!buff.empty() && buff != "\r") {
                if (currentCol < numcol) {
                    //std::cout << datas[currentCol].size() << ' ' << lineCount << buff << '\n';
                    datas[currentCol][lineCount] = (std::stof(buff));  // 将值存入当前列
                }
                buff.clear();
            }
            currentCol = currentCol + 1;
        }
        else if (ch == '\n') {
            // 遇到换行符，转换最后一个元素并重置列索引
            if (!buff.empty() && buff != "\r") {
                if (currentCol < numcol) {
                    datas[currentCol][lineCount] = (std::stof(buff));  // 将值存入当前列
                }
                buff.clear();
            }
            lineCount++;  // 增加行数
            currentCol = 0;  // 重置列索引以处理下一行
        }
        else {
            buff += ch;  // 累积字符到buff
        }
        count++;
    }

    // 如果最后一行没有换行符也要处理
    if (!buff.empty() && buff != "\r") {
        datas[currentCol][lineCount] = (std::stof(buff));
    }
    //std::cout << "readCount = " << readCount << std::endl;
    return datas;
}

void FFTfunction(std::vector<std::vector<float>> data, int numChannels) {
    //std::cout << picCount << '\n';
    picCount++;
    std::vector<torch::Tensor> data_tensors;
    data_tensors.push_back(torch::from_blob(data[0].data(), { static_cast<int64_t>(data[0].size()) }, torch::kFloat32));

    for (int i = 0; i < numChannels; i++) {
        //std::cout << i << '\n';
        data_tensors.push_back(torch::from_blob(data[i + 1].data(), { static_cast<int64_t>(data[i + 1].size()) }, torch::kFloat32));
    }

    if (useGPU) {
        moveDataToGPU(data_tensors, device);
    }
    time_t move_to_gpu = clock();

    for (int i = 1; i <= numChannels; ++i) {
        //std::cout << "picCount = "<< picCount << std::endl;
        float deltaT = data_tensors[0][1].item<float>() - data_tensors[0][0].item<float>();
        if (deltaT == 0) {
            std::cout << picCount << '\n';
            std::cerr << "Delta time is zero, invalid for frequency calculation." << std::endl;
            continue;
        }

        torch::Tensor time_tensor = torch::fft::fft(data_tensors[0].to(device));
        torch::Tensor fft_result = torch::fft::fft(data_tensors[i].to(device));
        torch::Tensor fft_freq = torch::fft::fftfreq(data_tensors[i].size(0), deltaT).to(device);
        time_t fft = clock();
        
         
        //fft_result = fft_result.to(torch::kCPU);
        //fft_freq = fft_freq.to(torch::kCPU);

        //moveDataToCPU(data_tensors);
        //time_t back_to_cpu = clock();

        //torch::Tensor fft_magnitude = torch::abs(fft_result);
        //std::vector<float> freq_vec(fft_freq.data_ptr<float>(), fft_freq.data_ptr<float>() + fft_freq.numel());
        //std::vector<float> magnitude_vec(fft_magnitude.data_ptr<float>(), fft_magnitude.data_ptr<float>() + fft_magnitude.numel());
        //time_t turn_to_vector = clock();

        //std::vector<float> print_freq, print_mag;
        //extractPositiveFreqAndMag(freq_vec, magnitude_vec, print_freq, print_mag);

        //time_t find_peak = clock();

        //int peak_index = find_peak_index(print_mag);
        //plotFFT(print_freq, print_mag, peak_index, "FFT of CH" + std::to_string(i), "b", i);
        //time_t show_figure = clock();

        //
        //std::cout << "time to move data to tensor: " << move_to_gpu - read_file << " milliseconds" << '\n';
        //std::cout << "time to compute fft: " << fft - move_to_gpu << " milliseconds" << '\n' << '\n';

        //std::cout << "time to move data back to cpu : " << back_to_cpu - fft << " milliseconds" << '\n';
        //std::cout << "time to turn data into vectors : " << turn_to_vector - back_to_cpu << " milliseconds" << '\n';
        //std::cout << "time to find peak : " << find_peak - turn_to_vector << " milliseconds" << '\n';
        //std::cout << "time to plot figure: " << show_figure - find_peak << " milliseconds" << "\n\n";
        //
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