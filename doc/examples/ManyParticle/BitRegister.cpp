#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("BitRegister");

//! [BitRegister]
#include "TBTK/BitRegister.h"
#include "TBTK/Streams.h"

using namespace TBTK;

int main(){
	BitRegister bitRegister[2];

	Streams::out << "Assignment\n";
	bitRegister[0] = 0xFF00FF00;
	Streams::out << bitRegister[0] << "\n";

	Streams::out << "\nSet individual bits\n";
	bitRegister[0].setBit(2, 1);
	bitRegister[0].setBit(30, 0);
	Streams::out << bitRegister[0] << "\n";

	bitRegister[0] = 0xFFFF0000;
	bitRegister[1] = 0x00FF00FF;

	Streams::out << "\nBitwise operations\n";
	Streams::out << (bitRegister[0] | bitRegister[1]) << "\n";
	Streams::out << (bitRegister[0] & bitRegister[1]) << "\n";
	Streams::out << (bitRegister[0] ^ bitRegister[1]) << "\n";

	Streams::out << "\nCount number of ones\n";
	Streams::out << bitRegister[0].getNumOneBits() << "\n";

	Streams::out << "\nClear a register\n";
	bitRegister[0].clear();
	Streams::out << bitRegister[0] << "\n";

	Streams::out << "\nBit shift\n";
	Streams::out << bitRegister[1] << "\n";
	Streams::out << (bitRegister[1] << 1) << "\n";
	Streams::out << (bitRegister[1] << 2) << "\n";
	Streams::out << (bitRegister[1] >> 1) << "\n";
	Streams::out << (bitRegister[1] >> 2) << "\n";
}
//! [BitRegister]
