#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("Resource");

//! [Resource]
#include "TBTK/Model.h"
#include "TBTK/Resource.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTK.h"

using namespace TBTK;

int main(){
	Initialize();

	//Write string to file.
	Resource resource;
	resource.setData("Hello quantum world!");
	resource.write("Message.txt");

	//Create a Model.
	Model model;
	model << HoppingAmplitude(-1, {0}, {1}) + HC;
	model.construct();

	//Write a serialized version of the Model to file.
	resource.setData(model.serialize(Serializable::Mode::JSON));
	resource.write("MyModel.json");

	//Read string message from file.
	resource.read("Message.txt");
	Streams::out << "Message:\n";
	Streams::out << resource.getData() << "\n\n";

	//Read Model from file.
	resource.read("MyModel.json");
	Model storedModel(resource.getData(), Serializable::Mode::JSON);
	Streams::out << "Stored Model:\n";
	Streams::out << storedModel << "\n";

	return 0;
}
//! [Resource]
