#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("Timer");

//! [Timer]
#include "TBTK/Timer.h"
#include "TBTK/Streams.h"

#include <complex>

using namespace std;
using namespace TBTK;

void functionA(){
	for(unsigned int n = 0; n < 100; n++)
		;
}

void functionB(){
	for(unsigned int n = 0; n < 200; n++)
		;
}

void timestampStack(){
	Timer::tick("Both functionA() and functionB()");

	Timer::tick("functionA()");
	functionA();
	Timer::tock();

	Timer::tick("functionB()");
	functionA();
	Timer::tock();

	Timer::tock();
}

void accumulators(){
	unsigned int idA = Timer::createAccumulator("functionA()");
	unsigned int idB = Timer::createAccumulator("functionB()");
	for(unsigned int n = 0; n < 100; n++){
		Timer::tick(idA);
		functionA();
		Timer::tock(idA);

		Timer::tick(idB);
		functionB();
		Timer::tock(idB);
	}
	Timer::printAccumulators();
}

int main(){
	timestampStack();
	accumulators();
}
//! [Timer]
