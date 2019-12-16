#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("Argument");

//! [Argument]
#include "TBTK/Streams.h"
#include "TBTK/TBTK.h"
#include "TBTK/Visualization/MatPlotLib/Argument.h"

using namespace std;
using namespace TBTK;
using namespace Visualization::MatPlotLib;

void print(const Argument &argument){
	const std::map<std::string, std::string> argumentMap
		= argument.getArgumentMap();
	if(argumentMap.size() == 0){
		Streams::out << "\t" << argument.getArgumentString() << "\n";
	}
	else{
		for(auto element : argumentMap){
			Streams::out
				<< "\t" << element.first
				<< ": " << element.second
				<< "\n";
		}
	}
}

int main(){
	Initialize();

	Streams::out << "Argument string:\n";
	print("Argument");

	Streams::out << "Argument map:\n";
	print({
		{"First key", "First value"},
		{"Second key", "Second value"},
		{"Third key", "Third value"}
	});
}
//! [Argument]
