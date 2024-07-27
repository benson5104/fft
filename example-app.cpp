#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <ctime>
#include <matplotlibcpp.h>

namespace plt = matplotlibcpp;

std::vector<std::vector<double>> read_csv(int numCol);
void FFTfunction(std::vector<std::vector<double>> data, int numChannels);
torch::Device getDevice(bool b);
void moveDataToGPU(std::vector<torch::Tensor>& data_tensors, torch::Device device);
void moveDataToCPU(std::vector<torch::Tensor>& data_tensors);
void extractPositiveFreqAndMag(const std::vector<float>& freq, const std::vector<float>& magnitude, std::vector<float>& pos_freq, std::vector<float>& pos_mag);
int find_peak_index(const std::vector<float>& magnitudes);
void plotFFT(const std::vector<float>& freq, const std::vector<float>& magnitude, int peak_index, const std::string& title, const std::string& color, int i);


int main() {
    int numChannels;
    std::cout << "Enter the number of channels (excluding time): ";
    std::cin >> numChannels;
    auto data = read_csv(numChannels + 1);

    FFTfunction(data, numChannels);
    return 0;
}


std::vector<std::vector<double>> read_csv(int numCol) {
    std::string inputFile;
    std::cout << "Please input the inputFile name (including csv): ";
    std::cin >> inputFile;
    std::ifstream file(inputFile);
    std::vector<std::vector<double>> data;
    std::string line, cell;

    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + inputFile);
    }

    // Find the label line
    bool startReading = false;


    while (std::getline(file, line)) {
        if (line.find("Label") != std::string::npos) {
            break;
        }
    }
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::vector<double> parsed_row;
        std::stringstream lineStream(line);
        while (std::getline(lineStream, cell, ',')) {
            try {
                parsed_row.push_back(std::stod(cell));
            }
            catch (const std::invalid_argument& e) {
                std::cerr << "Invalid data: " << cell << " in line: " << line << std::endl;
                throw;
            }
        }
        if (!parsed_row.empty()) {
            data.push_back(parsed_row);
        }
    }
    return data;
}


void FFTfunction(std::vector<std::vector<double>> data, int numChannels) {
    std::vector<float> time;
    std::vector<std::vector<float>> channels(numChannels);
    for (const auto& row : data) {
        time.push_back(static_cast<float>(row[0]));
        for (int i = 0; i < numChannels; ++i) {
            channels[i].push_back(static_cast<float>(row[i + 1]));
        }
    }
    time_t read_file = clock();

    std::cout << "please choose device (0: cpu, 1: gpu):  ";
    bool useGPU;
    std::cin >> useGPU;
    torch::Device device = getDevice(useGPU);

    std::vector<torch::Tensor> data_tensors;
    data_tensors.push_back(torch::from_blob(time.data(), { static_cast<int64_t>(time.size()) }, torch::kFloat32));

    for (auto& channel : channels) {
        data_tensors.push_back(torch::from_blob(channel.data(), { static_cast<int64_t>(channel.size()) }, torch::kFloat32));
    }

    if (useGPU)
    {
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

        fft_result = fft_result.to(torch::kCPU);
        fft_freq = fft_freq.to(torch::kCPU);

        moveDataToCPU(data_tensors);
        time_t back_to_cpu = clock();

        torch::Tensor fft_magnitude = torch::abs(fft_result);
        std::vector<float> freq_vec(fft_freq.data_ptr<float>(), fft_freq.data_ptr<float>() + fft_freq.numel());
        std::vector<float> magnitude_vec(fft_magnitude.data_ptr<float>(), fft_magnitude.data_ptr<float>() + fft_magnitude.numel());

        std::vector<float> print_freq, print_mag;
        extractPositiveFreqAndMag(freq_vec, magnitude_vec, print_freq, print_mag);
        int peak_index = find_peak_index(print_mag);

        plotFFT(print_freq, print_mag, peak_index, "FFT of CH" + std::to_string(i), "b", i);
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
    plt::figure();
    plt::plot(freq, magnitude, color + "-");
    plt::annotate("Peak (" + std::to_string(freq[peak_index]) + ", " + std::to_string(magnitude[peak_index]) + ")",
        freq[peak_index], magnitude[peak_index]);
    plt::title(title);
    plt::xlabel("Frequency (Hz)");
    plt::ylabel("Amplitude");

    std::string outputFile;
    std::cout << "Please input the outputFile name for " << i << ": ";
    std::cin >> outputFile;
    plt::save(outputFile);

    plt::show();
}
