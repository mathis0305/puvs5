// compile in Linux with gcc:
// g++ hello_world.cpp -lOpenCL

#include "CL/cl.h"                              // Verwendung der OpenCL Header Datei
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DATA_SIZE   10                          // Festlegen der Datengröße
#define MEM_SIZE    DATA_SIZE * sizeof(float)   // FEstelegen der Speichergröße

/** Erstellen der Kernelsource **/
const char* KernelSource =
"#define DATA_SIZE 10												\n"
"__kernel void test(__global float *input, __global float *output)  \n"
"{																	\n"
"	size_t i = get_global_id(0);									\n"
"	output[i] = input[i] * input[i];								\n"
"}																	\n"
"\n";

/** Deklaration der Variablen für das Hello World Programm**/
int main(void)
{
	cl_int				err;                      // Deklarieren des Error
	cl_platform_id* platforms = NULL;             // Nummer zur Definition der Plattform
	char			    platform_name[1024];      // Plattform Name
	cl_device_id	    device_id = NULL;         // Nummer zur Defintion des Geräts
	cl_uint			    num_of_platforms = 0,     // Anzahl der Plattformen
		num_of_devices = 0;						  // Anzahl der Geräte
	cl_context 			context;                  // Deklarieren des Kontexts
	cl_kernel 			kernel;                   // Deklarieren des Kernels
	cl_command_queue	command_queue;            // Deklarieren der Befehlswarteschlange
	cl_program 			program;                  // Deklarieren des Programms
	cl_mem				input, output;            // Deklarieren des Speichers für input und output
	float				data[DATA_SIZE] =         // Festlegen des Datensatzes
	{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	size_t				global[1] = { DATA_SIZE };  // Initialisierung der globalen Variable
	float				results[DATA_SIZE] = { 0 }; // Initialisierung der Ergebnisse

	/* 1) Initialisierung der Geräte*/

	// Abfrage welche Plattform gerade da ist
	err = clGetPlatformIDs(0, NULL, &num_of_platforms);
	if (err != CL_SUCCESS)
	{
		printf("No platforms found. Error: %d\n", err);
		return 0;
	}

	// Abfragen der Liste aller Plattformen 
	platforms = (cl_platform_id*)malloc(num_of_platforms);
	err = clGetPlatformIDs(num_of_platforms, platforms, NULL);
	if (err != CL_SUCCESS)
	{
		printf("No platforms found. Error: %d\n", err);
		return 0;
	}
	else
	{
		int nvidia_platform = 0;

		// Iterieren über alle Plattformen
		for (unsigned int i = 0; i < num_of_platforms; i++)
		{
			// Abbruch falls keine Plattformen mehr vorhanden sind
			clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
			if (err != CL_SUCCESS)
			{
				printf("Could not get information about platform. Error: %d\n", err);
				return 0;
			}

			// Erfolg wenn der Plattformname = NVIDIA ist
			if (strstr(platform_name, "NVIDIA") != NULL)
			{
				nvidia_platform = i;
				break;
			}
		}

		// GPU Device
		err = clGetDeviceIDs(platforms[nvidia_platform], CL_DEVICE_TYPE_GPU, 1, &device_id, &num_of_devices);
		if (err != CL_SUCCESS)
		{
			printf("Could not get device in platform. Error: %d\n", err);
			return 0;
		}
	}

	// Erzeugen eines OpenCl Kontexts
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("Unable to create context. Error: %d\n", err);
		return 0;
	}

	//  Erzeugen einer Befehlswarteschlange
	command_queue = clCreateCommandQueue(context, device_id, 0, &err);
	if (err != CL_SUCCESS)
	{
		printf("Unable to create command queue. Error: %d\n", err);
		return 0;
	}

	// Erstellen des Programms
	program = clCreateProgramWithSource(context, 1, (const char**)&KernelSource, NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("Unable to create program. Error: %d\n", err);
		return 0;
	}

	// Kompilieren und linken des Kernel-Quellexts
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error building program. Error: %d\n", err);
		return 0;
	}

	// Definieren des Kernel Einsprungspunktes
	kernel = clCreateKernel(program, "test", &err);
	if (err != CL_SUCCESS)
	{
		printf("Error setting kernel. Error: %d\n", err);
		return 0;
	}


	/* 2) Kontext und Befehlswarteschlange*/

	// Erzeuge Puffer für Ein- und Ausgabe
	input = clCreateBuffer(context, CL_MEM_READ_ONLY, MEM_SIZE, NULL, &err);
	output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, MEM_SIZE, NULL, &err);

	// Kopiere zusammenhängende Daten aus "data" in Eingabe-Puffer von input
	clEnqueueWriteBuffer(command_queue, input, CL_TRUE, 0, MEM_SIZE, data, 0, NULL, NULL);

	// Definiere die Reihenfolge der Argumente des Kerns
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);


	/* 3)  Speicher für Ein- und Ausgabe zuweisen*/

	// Einreihen des Kerns in die Befehlswarteschlange und AUfteilungsbreich angeben
	clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global, NULL, 0, NULL, NULL);

	// Auf die Beendigung der Operation warten
	clFinish(command_queue);

	// Kopiere die Ergebnisse vom Ausgabe-Puffer "output" in das Ergebnisfeld "result"
	clEnqueueReadBuffer(command_queue, output, CL_TRUE, 0, MEM_SIZE, results, 0, NULL, NULL);

	// Ausgabe der Ergebnisse
	for (unsigned int i = 0; i < DATA_SIZE; i++)
		printf("%f\n", results[i]);


	/* 4) Aufräumen der OpenCL Ressourcen*/
	clReleaseMemObject(input);
	clReleaseMemObject(output);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	return 0;
}
