#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("ExtensiveBitRegister");

//! [ExtensiveBitRegister]
#include "TBTK/ExtensiveBitRegister.h"
#include "TBTK/Streams.h"

using namespace TBTK;

int main(){
	ExtensiveBitRegister bitRegister0(64);
	ExtensiveBitRegister bitRegister1(64);

	Streams::out << "Assignment\n";
	bitRegister0 = 0xFF00FF00;
	Streams::out << bitRegister0 << "\n";

	Streams::out << "\nSet individual bits\n";
	bitRegister0.setBit(2, 1);
	bitRegister0.setBit(30, 0);
	Streams::out << bitRegister0 << "\n";

	bitRegister0 = 0xFFFF0000;
	bitRegister1 = 0x00FF00FF;

	Streams::out << "\nBitwise operations\n";
	Streams::out << (bitRegister0 | bitRegister1) << "\n";
	Streams::out << (bitRegister0 & bitRegister1) << "\n";
	Streams::out << (bitRegister0 ^ bitRegister1) << "\n";

	Streams::out << "\nCount number of ones\n";
	Streams::out << bitRegister0.getNumOneBits() << "\n";

	Streams::out << "\nClear a register\n";
	bitRegister0.clear();
	Streams::out << bitRegister0 << "\n";

	Streams::out << "\nBit shift\n";
	Streams::out << bitRegister1 << "\n";
	Streams::out << (bitRegister1 << 1) << "\n";
	Streams::out << (bitRegister1 << 2) << "\n";
	Streams::out << (bitRegister1 >> 1) << "\n";
	Streams::out << (bitRegister1 >> 2) << "\n";
}
//! [ExtensiveBitRegister]
