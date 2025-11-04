#include "../src/cudamat.h"
#include <stdio.h>

int main()
{
    Matrix a = init_rand(3, 3);
    Matrix b = init_rand(3, 3);

    mprint(a);
    mprint(b);

    Matrix c = madd(a, b);

    mprint(c);

    mfree(a);
    mfree(b);
    mfree(c);
}
