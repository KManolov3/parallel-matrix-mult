#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <err.h>
#include <sys/wait.h>
#define TIMES_TO_EXEC 5
#define MAX_THREADS 16

const char OUT_FILE_NAME[] = "measurements.txt";
const char AVG_OUT_FILE_NAME[] = "averaged_measurements.txt";

void measureProgram(char* argv[]){     
    char* programPath = argv[1];
    int fdMeasurements;
    if((fdMeasurements = open(OUT_FILE_NAME, O_WRONLY | O_CREAT | O_TRUNC, S_IRWXU)) == -1){
        err(1, "Couldn't open file");
    }
    
    int pid;
    for(int i=1; i<=MAX_THREADS; i++){
        char iToStr[3]="";
        sprintf(iToStr, "%d", i);
        for(int j=0; j<TIMES_TO_EXEC; j++){
            pid = fork();
            if(pid == -1){
                err(2, "Couldn't fork");
            }
            if(pid == 0){
                dup2(fdMeasurements, 1);
                execl(programPath, programPath, "-m", argv[2], "-n", argv[3], "-k", argv[4], "-t", iToStr, "-q", NULL);
            }
            wait(NULL);
        }
    }
    close(fdMeasurements);       
}

void averageMeasurements(){
    FILE *measurements, *avgMeasurements;
    measurements = fopen(OUT_FILE_NAME, "r");
    avgMeasurements = fopen(AVG_OUT_FILE_NAME, "w");
    double totalParallel, curParallel;
    for(int i=1; i<MAX_THREADS; i++){
        totalParallel=0;
        for(int j=0; j<TIMES_TO_EXEC; j++){
            fscanf(measurements, "%lf", &curParallel);
            totalParallel += curParallel;
        }
        fprintf(avgMeasurements, "%lf\n", totalParallel/TIMES_TO_EXEC);
    }
}

int main(int argc, char* argv[]){
    if(argc<=4){
        errx(1, "Please provide path and input size for program to be measured");  
    }
    
    measureProgram(argv);
    averageMeasurements();

    exit(0);

}    
