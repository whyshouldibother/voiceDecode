#include <vector>
#include<iostream>
int main(){
    std::vector<int> squares(1000);
    #pragma omp parallel for
    for(int i=0; i<1000; i++){
        squares[i]=i*i;
        std::cout<<i<<squares[i]<<'\n';
    }
}
