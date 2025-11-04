#include "../src/cudamat.h"
#include <stdio.h>

int main()
{
    Matrix a = init_rand(10000, 100000);
    // Matrix b = init_rand(10000, 10000);

    //mprint(a);
    // mprint(b);

    // Matrix c = madd(a, b);

    // mprint(c);
    printf("Done!\n");

    mfree(a);
    // mfree(b);
    // mfree(c);
    unhandle();
}
