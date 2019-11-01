#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("Communicator");

//! [Communicator]
#include "TBTK/Communicator.h"
#include "TBTK/Streams.h"

#include <vector>

using namespace std;
using namespace TBTK;

class MyCommunicator : public Communicator{
public:
	MyCommunicator() : Communicator(true){}

	void say(const std::string &message){
		if(getVerbose() && Communicator::getGlobalVerbose()){
			Streams::out << message << "\n";
		}
	}
};

int main(){
	MyCommunicator myCommunicator;
	myCommunicator.setVerbose(false);
	myCommunicator.say("Don't say this");

	myCommunicator.setVerbose(true);
	myCommunicator.say("Say this");

	Communicator::setGlobalVerbose(false);
	myCommunicator.say("Don't say this");
}
//! [Communicator]
