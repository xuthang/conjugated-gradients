#include "../solver.cu"
struct TIMER
{
    std::function<void(double)> f;
    std::chrono::high_resolution_clock::time_point begin;

    TIMER(std::function<void(double)> func = [](double res){std::cout << res << std::endl;})
        : f(func), begin(std::chrono::high_resolution_clock::now()) {}

    ~TIMER()
    {
        auto end = std::chrono::high_resolution_clock::now();
        double result = (std::chrono::duration_cast<std::chrono::microseconds >(end - begin).count() / 1000.);
        f(result);
    }
};


int main()
{
    int n = 2000000;
    const int threadCnt = 512;
    const int blocksCnt = n / threadCnt + (n % threadCnt != 0);
    host_vector<cVectorType> aHost(n), bHost(n);
    for(int i = 0; i < n; i++)
    {
        aHost[i] = (rand() % 100);
        bHost[i] = (rand() % 100);
    }

    {
        TIMER T;
        cout << inner_product(aHost.begin(), aHost.end(), bHost.begin(), (cVectorType)0) << endl;
    }


    device_vector<cVectorType> a = aHost, b = bHost;
    device_vector<cVectorType> ret(1);

    {
        TIMER t;
        cudaVectorMult<<<blocksCnt, threadCnt>>>(a.data().get(), b.data().get(), ret.data().get(), n);
        CUDA_CHECK(cudaDeviceSynchronize());
        cout << ret[0] << endl;
    }
    return 0;
}