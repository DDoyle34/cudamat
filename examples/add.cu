#include "../src/cudamat.h"
#include <stdio.h>

int main()
{
    Matrix a = init_rand(3, 3);
    Matrix b = init_const(3, 3, 4);

    mprint(a);
    mprint(b);

    Matrix c = madd(a, b);

    mprint(c);
    printf("Done!\n");

    mfree(a);
    mfree(b);
    mfree(c);
    unhandle();
}
