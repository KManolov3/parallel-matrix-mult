#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <sys/time.h>
#include <pthread.h>
#include <omp.h>
#include <math.h>
#include <string.h>
#include <err.h>
#define MICRO 0.000001
typedef double** Matrix;

const int DIM = 1024;


double** emptyMatrix(int rows, int cols){
    double** matrix = malloc(rows * sizeof(double*));    
    
    for(int i=0; i<rows; i++){
        matrix[i] = malloc(cols * sizeof(double));
    }

    return matrix;
}

double** randomMatrix(int rows, int cols){
    double** matrix = emptyMatrix(rows, cols);

    srand(time(NULL)+clock()+rand());

    #pragma omp parallel for
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            matrix[i][j] = rand();
        }
    }

    return matrix;
}

void readFileDimensions(FILE* file, int *m, int *n, int *k){
    if(!fscanf(file, "%d", m) || !fscanf(file, "%d", n) || !fscanf(file, "%d", k)){
        pclose(file);
        errx(3, "Incorrect input file format");
    }
}

void readMatrix(FILE* file, Matrix matrix, int rows, int cols){
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            if(!fscanf(file, "%lf", &matrix[i][j])){
                errx(3, "Incorrect input file format");
            }
        }
    }
}

void readFile(FILE* file, int *m, int *n, int *k, Matrix *matrixA, Matrix *matrixB){
    readFileDimensions(file, m, n, k);
    if(*m<=0 || *n<=0 || *k<=0){
        errx(2, "Incorrect file format! m, n and k must be valid positive numbers\n");
    }
    *matrixA = emptyMatrix(*m,*n);
    *matrixB = emptyMatrix(*n,*k);
    readMatrix(file, *matrixA, *m, *n);
    readMatrix(file, *matrixB, *n, *k);
     
}

void writeToFile(FILE* file, Matrix matrix, int m, int k){
    fprintf(file, "%d %d\n", m, k);

    for(int i=0; i<m; i++){
        for(int j=0; j<k; j++){
            fprintf(file, "%.2lf ", matrix[i][j]);
        }
        fprintf(file, "\n");
    }
}

void help(){
    printf("OPTIONS:\n");
    printf("\t-m <num rows>: number of rows of first matrix\n");
    printf("\t-n <num rows/cols>: shared dimension between matrices\n");
    printf("\t-k <num cols>: number of cols of second matrix\n");
    printf("\t-i <filename>, --input-file <filename>: input file name\n");
    printf("\t-o <filename>, --output-file <filename>: output file name\n");
    printf("\t-t <num threads>, --threads <num threads>: number of concurrent threads to be used\n");
    printf("\t-q, --quiet: quiet mode; outputs only the execution time of the parallel version, followed by a newline\n");
}

double* flattenSequential(Matrix matrix, int rows, int cols, bool columnwise){
    double* flatMatrix = malloc(rows * cols * sizeof(double));
 
    if(columnwise){
        for(int i=0; i<rows; i++){
            for(int j=0; j<cols; j++){
                flatMatrix[j*rows + i] = matrix[i][j];
            }
        }
    } else {
        for(int i=0; i<rows; i++){
            int offset = i*cols;
            for(int j=0; j<cols; j++){
                flatMatrix[offset + j] = matrix[i][j];
            }
        }
    }

    return flatMatrix;
}    


double multiplySequential(Matrix matrixA, Matrix matrixB, Matrix matrixC, int rowsA, int colsA /*==rowsB*/, int colsB){
    struct timeval time;
    double startTime, endTime;
    
    gettimeofday(&time, NULL);
    startTime = time.tv_sec + MICRO * time.tv_usec;

    double* flatA = flattenSequential(matrixA, rowsA, colsA, false);
    double* flatB = flattenSequential(matrixB, colsA, colsB, true);

    int aOffset, bOffset, i, j, k;
    double total;
    for(i=0; i<rowsA; i++){
        aOffset = i*colsA;
        for(j=0; j<colsB; j++){
            bOffset = j*colsA;
            total = 0;
            for(k=0; k<colsA; k++){
                total += flatA[aOffset + k] * flatB[bOffset + k];
            }
         matrixC[i][j] = total;
        }
    }

    gettimeofday(&time, NULL);
    endTime = time.tv_sec + MICRO * time.tv_usec;
    double elapsedTime = endTime - startTime;

    free(flatA);
    free(flatB);

    return elapsedTime;
}


double* flattenParallel(Matrix matrix, int rows, int cols, bool columnWise){
    double* flatMatrix = malloc(rows * cols * sizeof(double));
    
    if(columnWise){
        #pragma omp parallel for
        for(int i=0; i<rows; i++){
            for(int j=0; j<cols; j++){
                flatMatrix[j*rows + i] = matrix[i][j];
            }
        }
    } else {
        #pragma omp parallel for
        for(int i=0; i<rows; i++){
            int offset = i*cols;
            for(int j=0; j<cols; j++){
                flatMatrix[offset + j] = matrix[i][j];
            }
        }
    }

    return flatMatrix;
}    

double multiplyParallel(Matrix matrixA, Matrix matrixB, Matrix matrixC, int rowsA, int colsA /*==rowsB*/, int colsB, int num_threads, bool isQuiet){
    omp_set_num_threads(num_threads);
        
    struct timeval time;
    double startTime, endTime;
    
    gettimeofday(&time, NULL);
    startTime = time.tv_sec + MICRO * time.tv_usec;

    double* flatA = flattenSequential(matrixA, rowsA, colsA, false);
    double* flatB = flattenSequential(matrixB, colsA, colsB, true);
    

    int aOffset, bOffset, i, j, k;
    double total;
    #pragma omp parallel shared(matrixC) private(i, j, k, aOffset, bOffset, total) 
    {
        double wtime = omp_get_wtime();
        #pragma omp for schedule(static)
        for(i=0; i<rowsA; i++){
		aOffset = i*colsA;
		for(j=0; j<colsB; j++){
		    bOffset = j*colsA;
		    total = 0;
                    for(k=0; k<colsA; k++){
		        total += flatA[aOffset + k] * flatB[bOffset + k];
		    }
		 matrixC[i][j] = total;
        	}
   	 }
        wtime = omp_get_wtime() - wtime;
        if(!isQuiet){
            printf( "Time taken by thread %d is %f\n", omp_get_thread_num(), wtime );
        }
    }

    gettimeofday(&time, NULL);
    endTime = time.tv_sec + MICRO * time.tv_usec;
    double elapsedTime = endTime - startTime;

    free(flatA);
    free(flatB);

    return elapsedTime;
}

bool checkIfEqual(Matrix m1, Matrix m2, int rows, int cols){
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            if(m1[i][j] != m2[i][j]){
                return false;
            }
        }
    }
    
    return true;
}

void freeMatrix(Matrix m, int rows){
    for(int i=0; i<rows; i++){
        free(m[i]);
    }
    free(m);
}

int main(int argc, char* argv[])
{
    int m=0, n=0, k=0, numThreads=1;
    char* inputFile=0, *outputFile=0;
    bool isQuiet = false;
    for(int i=1; i<argc; i++){
        if(strcmp(argv[i], "-m") == 0){
            if(i+1 >= argc){
                errx(6, "Incorrect program usage");
            }
            m = atoi(argv[i+1]);
            i++;
            continue;
        }
        if(strcmp(argv[i], "-n") == 0){ 
            if(i+1 >= argc){
                errx(6, "Incorrect program usage");
            }
            n = atoi(argv[i+1]);
            i++;
            continue;
        }
        if(strcmp(argv[i], "-k") == 0){
            if(i+1 >= argc){
                errx(6, "Incorrect program usage");
            }
            k = atoi(argv[i+1]);
            i++;
            continue;
        }
        if(!strcmp(argv[i], "-t") || !strcmp(argv[i], "--threads")){
            if(i+1 >= argc){
                errx(6, "Incorrect program usage");
            }
            numThreads = atoi(argv[i+1]);
            i++;
            continue;
        }
        if(!strcmp(argv[i], "-i") || !strcmp(argv[i], "--input-file")){
            if(i+1 >= argc){
                errx(6, "Incorrect program usage");
            }
            inputFile = argv[i+1];
            i++;
            continue;
        }
        if(!strcmp(argv[i], "-o") || !strcmp(argv[i], "--output-file")){
            if(i+1 >= argc){
                errx(6, "Incorrect program usage");
            }
            outputFile = argv[i+1];
            i++;
            continue;
        }
        if(!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")){
            help();
            exit(0);
        }
        if(!strcmp(argv[i], "-q") || !strcmp(argv[i], "--quiet")){
            isQuiet = true;
        }
    }
    
    Matrix matrixA=0, matrixB=0;    

    if(argc<=1){
        help();
        exit(0);
    }

    if(inputFile == NULL){
        if(m<=0 || n<=0 || k<=0){
            help();
            errx(1, "Incorrect program usage! m, n and k must all be set to positive integers! Please provide valid input arguments!");
        }
        matrixA = randomMatrix(m,n);
        matrixB = randomMatrix(n,k);
    } else {
        FILE *file = fopen(inputFile, "r");
        if(file == NULL){
            err(4, "Invalid input file argument!");
        }        
        readFile(file, &m, &n, &k, &matrixA, &matrixB);
        fclose(file);
    }    
    Matrix matrixC = emptyMatrix(m,k),
           verificationMatrix = emptyMatrix(m,k);
    double elapsedTime, elapsedTimeParallel; 

    if(!isQuiet){
        elapsedTime = multiplySequential(matrixA, matrixB, verificationMatrix, m, n, k);
        printf("Time it took for multiplication was %lf seconds\n", elapsedTime);
        elapsedTimeParallel = multiplyParallel(matrixA, matrixB, matrixC, m, n, k, numThreads, isQuiet);    
        printf("Time it took for parallel multiplication was %lf seconds\n", elapsedTimeParallel);
    
        double acceleration = elapsedTime/elapsedTimeParallel;
        printf("The parallel multiplication of matrices with dimensions %d,%d and %d,%d using %d threads was %lf times faster than the serial\n", m, n, n, k, numThreads, acceleration); 

        bool areMatricesEqual = checkIfEqual(matrixC, verificationMatrix, m, k);
        areMatricesEqual ? printf("The result matrices are equal\n") : printf("The result matrices are not equal\n");
    } else {
        elapsedTimeParallel = multiplyParallel(matrixA, matrixB, matrixC, m, n, k, numThreads, isQuiet);    
        printf("%lf\n", elapsedTimeParallel);
    }   

    if(outputFile != NULL){
        FILE *file = fopen(outputFile, "w");
        if(file == NULL){
            err(7, "Couldn't open output file");
        }
        writeToFile(file, matrixC, m, k);
        fclose(file);
    }

    freeMatrix(matrixA, m);
    freeMatrix(matrixB, n);
    freeMatrix(matrixC, m);
    freeMatrix(verificationMatrix, m);    
    return 0;
}
