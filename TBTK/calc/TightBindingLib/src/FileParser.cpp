/** @file FileParser.cpp
 *
 *  @author Kristofer Björnson
 *  @author Andreas Theiler
 */

#include "../include/FileParser.h"
#include "../include/Geometry.h"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

using namespace std;

namespace TBTK{

ofstream FileParser::fout;
stringstream FileParser::ssin;

void FileParser::writeModel(
	Model *model,
	string fileName,
	AmplitudeMode amplitudeMode,
	string description
){
	openOutput(fileName);

	writeDescription(description);

	writeAmplitudes(model, amplitudeMode);
	writeLineBreaks(1);

	writeGeometry(model);

	closeOutput();
}

Model* FileParser::readModel(string fileName){
	Model *model = new Model();

	readInput(fileName);
	removeComments();
	removeInitialWhiteSpaces();

	readAmplitudes(model);

	model->construct();

	readGeometry(model);

	return model;
}

Util::ParameterSet* FileParser::readParameterSet(std::string fileName){
	Util::ParameterSet *parameterSet = new Util::ParameterSet();

	readInput(fileName);
	removeComments();
	removeInitialWhiteSpaces();

	while(true){
		string line;

		if(ssin.peek() == EOF)
			break;
		if(ssin.peek() == '\n'){
			getline(ssin, line);
			continue;
		}

		string type;
		ssin >> type;
		if(type.compare("int") == 0){
			string name;
			ssin >> name;
			char equalitySign;
			ssin >> equalitySign;
			if(equalitySign != '='){
				cout << "Error in FileParser::readParameterSet(): Expected '=' but found '" << equalitySign << "'.\n";
				exit(1);
			}
			int value;
			ssin >> value;
			parameterSet->addInt(name, value);
		}
		else if(type.compare("double") == 0){
			string name;
			ssin >> name;
			char equalitySign;
			ssin >> equalitySign;
			if(equalitySign != '='){
				cout << "Error in FileParser::readParameterSet(): Expected '=' but found '" << equalitySign << "'.\n";
				exit(1);
			}
			double value;
			ssin >> value;
			parameterSet->addDouble(name, value);
		}
		else if(type.compare("complex") == 0){
			string name;
			ssin >> name;
			char equalitySign;
			ssin >> equalitySign;
			if(equalitySign != '='){
				cout << "Error in FileParser::readParameterSet(): Expected '=' but found '" << equalitySign << "'.\n";
				exit(1);
			}
			complex<double> value;
			ssin >> value;
			parameterSet->addComplex(name, value);
		}
		else if(type.compare("string") == 0){
			string name;
			ssin >> name;
			char equalitySign;
			ssin >> equalitySign;
			if(equalitySign != '='){
				cout << "Error in FileParser::readParameterSet(): Expected '=' but found '" << equalitySign << "'.\n";
				exit(1);
			}
			string value;
            getline(ssin, value);
            int first = value.find_first_not_of(" \t");
            int last = value.find_last_not_of(" \t");
            value = value.substr(first, last - first + 1);
			parameterSet->addString(name, value);
		}
        else if(type.compare("bool") == 0){
			string name;
			ssin >> name;
			char equalitySign;
			ssin >> equalitySign;
			if(equalitySign != '='){
				cout << "Error in FileParser::readParameterSet(): Expected '=' but found '" << equalitySign << "'.\n";
				exit(1);
			}
			bool value;
			ssin >> value;
			parameterSet->addBool(name, value);
		}
		else{
			cout << "Error in FileParser::readParametersSet(): Expected type but found '" << type << "'.\n";
			exit(1);
		}
/*		if(!getline(ssin, line)){
//			cout << "Error in FileParser::readAmplitudes(): Reached end of file while searching for 'Amplitudes:'.\n";
			exit(1);
		}

		unsigned int pos = line.find();
		if(line.find("Amplitudes:") != string::npos){
			int mode = readParameter("Mode", "Amplitude");
			if(mode < 0 || mode > 1){
				cout << "Error in FileParser::readAmplitudes(): Only Amplitude mode 0 and 1 supported.\n";
				exit(1);
			}
			amplitudeMode = static_cast<AmplitudeMode>(mode);

			break;
		}*/
	}

	return parameterSet;
}

void FileParser::openOutput(string fileName){
	fout.open(fileName);
}

void FileParser::closeOutput(){
	fout.close();
}

void FileParser::readInput(string fileName){
	fstream fin;
	fin.open(fileName);
	if(!fin){
		cout << "Error in FileParser::readInput(): Unable to open '" << fileName << "'.\n";
		exit(1);
	}

	ssin << fin.rdbuf();

	fin.close();
}

void FileParser::writeLineBreaks(int numLineBreaks){
	for(int n = 0; n < numLineBreaks; n++)
		fout << "\n";
}

void FileParser::writeTabs(int numTabs){
	for(int n = 0; n < numTabs; n++)
		fout << "\t";
}

void FileParser::write(complex<double> value){
	stringstream ss;
	ss << left << setw(10) << real(value) << setw(10) << imag(value);

	fout << ss.str();
}

void FileParser::write(const Index &index){
	stringstream ss;
	ss << "[";
	for(unsigned int n = 0; n < index.size(); n++){
		if(n != 0)
			ss << " ";
		ss << index.at(n);
	}
	ss << "]";

	fout << ss.str();
}

void FileParser::writeCoordinates(
	const double *coordinates,
	int numCoordinates
){
	stringstream ss;
	ss << "(";
	for(int n = 0; n < numCoordinates; n++){
		if(n != 0)
			ss << " ";
		ss << coordinates[n];
	}
	ss << ")";

	fout << ss.str();
}

void FileParser::writeSpecifiers(const int *specifiers, int numSpecifiers){
	stringstream ss;
	ss << "<";
	for(int n = 0; n < numSpecifiers; n++){
		if(n != 0)
			ss << " ";
		ss << specifiers[n];
	}
	ss << ">";

	fout << ss.str();
}

void FileParser::writeDescription(string description){
	if(description.size() == 0)
		return;

	stringstream ss;
	ss.str(description);
	string word;
	fout << "/*";
	int charCount = 2;
	while(ss >> word){
		charCount += word.size() + 2;
		if(charCount > 80){
			fout << "\n *";
			charCount = 2;
		}
		fout << " " << word;
	}
	if(charCount + 3 < 80)
		fout << " */";
	else
		fout << "\n */";

	fout << "\n\n";
}

void FileParser::writeAmplitudes(Model *model, AmplitudeMode amplitudeMode){
	fout << "Amplitudes:\n";
	fout << left << setw(30) << "Mode" << "= " << static_cast<int>(amplitudeMode) << "\n";

	AmplitudeSet::Iterator it = model->getAmplitudeSet()->getIterator();
	HoppingAmplitude *ha;
	while((ha = it.getHA())){
		switch(amplitudeMode){
		case AmplitudeMode::ALL:
			fout << left << setw(30);
			write(ha->getAmplitude());
			fout << left << setw(20);
			write(ha->toIndex);
			write(ha->fromIndex);
			writeLineBreaks(1);
			break;
		case AmplitudeMode::ALL_EXCEPT_HC:
		{
			int from = model->getBasisIndex(ha->fromIndex);
			int to = model->getBasisIndex(ha->toIndex);
			if(from <= to){
				fout << left << setw(30);
				write(ha->getAmplitude());
				fout << setw(20);
				write(ha->toIndex);
				write(ha->fromIndex);
				writeLineBreaks(1);
			}
			break;
		}
		default:
			cout << "Eror in FileParser::writeAmplitudes: Unsupported amplitudeMode (" << static_cast<int>(amplitudeMode) << ").\n";
			exit(1);
		}

		it.searchNextHA();
	}
}

void FileParser::writeGeometry(Model *model){
	Geometry *geometry = model->getGeometry();

	if(geometry == NULL){
		fout << "Geometry: None\n";
		return;
	}
	else{
		fout << "Geometry:\n";
	}

	int dimensions = geometry->getDimensions();
	const int numSpecifiers = geometry->getNumSpecifiers();
	fout << left << setw(30) << "Dimensions" << "= " << dimensions << "\n";
	fout << left << setw(30) << "Num specifiers" << "= " << numSpecifiers << "\n";

	AmplitudeSet::Iterator it = model->getAmplitudeSet()->getIterator();
	HoppingAmplitude *ha;
	Index dummyIndex({-1});
	Index &prevIndex = dummyIndex;//Start with dummy index
	while((ha = it.getHA())){
		Index &index = ha->fromIndex;
		if(!index.equals(prevIndex)){
			const double *coordinates = geometry->getCoordinates(index);
			const int *specifiers = geometry->getSpecifiers(index);
			fout << left << setw(30);
			writeCoordinates(coordinates, dimensions);
			fout << setw(20);
			writeSpecifiers(specifiers, numSpecifiers);
			write(index);
			writeLineBreaks(1);

			prevIndex = index;
		}

		it.searchNextHA();
	}
}

void FileParser::removeComments(){
	stringstream sstemp;

	const int STATE_NORMAL = 0;
	const int STATE_SLASH_FOUND = 1;
	const int STATE_SINGLE_LINE_COMMENT = 2;
	const int STATE_MULTI_LINE_COMMENT = 3;
	const int STATE_MULTI_LINE_COMMENT_ASTERIX_FOUND = 4;

	int state = STATE_NORMAL;
	char c;
	ssin >> noskipws;
	while(ssin >> c){
		switch(state){
		case STATE_NORMAL:
			switch(c){
			case '/':
				state = STATE_SLASH_FOUND;
				break;
			default:
				sstemp << c;
				break;
			}
			break;
		case STATE_SLASH_FOUND:
			switch(c){
			case '/':
				state = STATE_SINGLE_LINE_COMMENT;
				break;
			case '*':
				state = STATE_MULTI_LINE_COMMENT;
				break;
			default:
				state = STATE_NORMAL;
				sstemp << '/' << c;
				break;
			}
			break;
		case STATE_SINGLE_LINE_COMMENT:
			switch(c){
			case '\n':
				state = STATE_NORMAL;
				sstemp << "\n";
				break;
			default:
				break;
			}
			break;
		case STATE_MULTI_LINE_COMMENT:
			switch(c){
			case '*':
				state = STATE_MULTI_LINE_COMMENT_ASTERIX_FOUND;
				break;
			default:
				break;
			}
			break;
		case STATE_MULTI_LINE_COMMENT_ASTERIX_FOUND:
			switch(c){
			case '/':
				state = STATE_NORMAL;
				break;
			default:
				break;
			}
			break;
		default:
			cout << "Error in FileParser::removeComments(): Unknown state.\n";
			exit(1);
		}
	}

	ssin.str("");
	ssin.clear();
	ssin << sstemp.rdbuf();
	ssin >> skipws;
}

void FileParser::removeInitialWhiteSpaces(){
	stringstream sstemp;
	ssin >> noskipws;

	const int STATE_FIRST_NWS_CHARACTER_NOT_FOUND = 0;
	const int STATE_FIRST_NWS_CHARACTER_FOUND = 1;
	int state = STATE_FIRST_NWS_CHARACTER_NOT_FOUND;
	char c;
	while(ssin >> c){
		switch(state){
		case STATE_FIRST_NWS_CHARACTER_NOT_FOUND:
			switch(c){
			case ' ':
				break;
			case '\t':
				break;
			case '\n':
				sstemp << c;
				break;
			default:
				state = STATE_FIRST_NWS_CHARACTER_FOUND;
				sstemp << c;
				break;
			}
			break;
		case STATE_FIRST_NWS_CHARACTER_FOUND:
			switch(c){
			case '\n':
				state = STATE_FIRST_NWS_CHARACTER_NOT_FOUND;
				sstemp << c;
				break;
			default:
				sstemp << c;
				break;
			}
			break;
		default:
			cout << "Error in FileParser::removeInitialWhiteSpaces(): Unknown state.\n";
			exit(1);
		}
	}

	ssin.str("");
	ssin.clear();
	ssin << sstemp.rdbuf();
	ssin >> skipws;
}

void FileParser::readAmplitudes(Model *model){
	AmplitudeMode amplitudeMode;
	string line;
	while(true){
		if(!getline(ssin, line)){
			cout << "Error in FileParser::readAmplitudes(): Reached end of file while searching for 'Amplitudes:'.\n";
			exit(1);
		}

		if(line.find("Amplitudes:") != string::npos){
			int mode = readParameter("Mode", "Amplitude");
			if(mode < 0 || mode > 1){
				cout << "Error in FileParser::readAmplitudes(): Only Amplitude mode 0 and 1 supported.\n";
				exit(1);
			}
			amplitudeMode = static_cast<AmplitudeMode>(mode);

			break;
		}
	}

	while(true){
		if(ssin.peek() == '\n' || ssin.peek() == EOF)
			break;

		HoppingAmplitude *ha = readHoppingAmplitude();
		if(ha == NULL)
			break;

		switch(amplitudeMode){
		case AmplitudeMode::ALL:
			model->addHA(*ha);
			break;
		case AmplitudeMode::ALL_EXCEPT_HC:
		{
			const Index &from = ha->fromIndex;
			const Index &to = ha->toIndex;
			if(from.equals(to))
				model->addHA(*ha);
			else
				model->addHAAndHC(*ha);
			break;
		}
		case AmplitudeMode::UNIT_CELL:
			//To be implemented.
			break;
		case AmplitudeMode::UNIT_CELL_EXCEPT_HC:
			//To be implemented.
			break;
		}

		delete ha;

		//Throw away end of line
		string line;
		getline(ssin, line);
	}
}

void FileParser::readGeometry(Model *model){
	int dimensions;
	int numSpecifiers;
	string line;
	while(true){
		if(!getline(ssin, line)){
			cout << "Warning in FileParser::readAmplitudes(): Reached end of file while searching for 'Geometry:'.\n";
			cout << "\tNo Geometry loaded.\n";
			cout << "\tAdd 'Geometry: None' after amplitude list to disable warning.\n";
			return;
		}

		if(line.find("Geometry:") != string::npos){
			if(line.find("None") != string::npos)
				return;

			dimensions = readParameter("Dimensions", "Geometry");
			numSpecifiers = readParameter("Num specifiers", "Geometry");
			cout << dimensions << " " << numSpecifiers << "\n";

			break;
		}
	}

	model->createGeometry(dimensions, numSpecifiers);
	Geometry *geometry = model->getGeometry();

	while(true){
		if(ssin.peek() == '\n' || ssin.peek() == EOF)
			break;

		vector<double> coordinates;
		readCoordinates(&coordinates, dimensions);

		vector<int> specifiers;
		if(numSpecifiers > 0){
			readSpecifiers(&specifiers, numSpecifiers);
		}
		else{
			//Clear empty specifiers
			char c;
			ssin >> c;
			ssin >> c;
		}

		Index *index = readIndex();

		geometry->setCoordinates(*index, coordinates, specifiers);

		delete index;

		//Throw away end of line
		string line;
		getline(ssin, line);
	}
}

int FileParser::readParameter(
	const string parameterName,
	const string parentStructure
){
	string line;

	if(!getline(ssin, line)){
		cout << "Error in FileParser::readParameter(): Expected parameter '" << parameterName << "' for structure '" << parentStructure << "'.\n";
		exit(1);
	}

	size_t position = line.find(parameterName);
	if(position == string::npos){
		cout << "Error in FileParser::readAmplitudes(): Expected parameter '" << parameterName << "' for structure '" << parentStructure << "'.\n";
		exit(1);
	}
	line = line.substr(position+4);

	position = line.find("=");
	if(position == string::npos){
		cout << "Error in FileParser::readAmplitudes(): Expected '=' after " << parameterName << ".\n";
		exit(1);
	}
	line = line.substr(position+1);

	stringstream ss;
	ss << line;
	int value;
	ss >> value;

	return value;
}

HoppingAmplitude* FileParser::readHoppingAmplitude(){
	complex<double> amplitude;
	if(readComplex(&amplitude))
		return NULL;

	Index *to = readIndex();
	if(to == NULL){
		cout << "Error in FileParser::readHoppingAmplitude(): Faile to read index.\n";
		exit(1);
	}

	Index *from = readIndex();
	if(to == NULL){
		cout << "Error in FileParser::readHoppingAmplitude(): Faile to read index.\n";
		exit(1);
	}

	HoppingAmplitude *ha = new HoppingAmplitude(amplitude, *to, *from);
	delete to;
	delete from;

	return ha;
}

Index* FileParser::readIndex(){
	char c;
	ssin >> c;
	if(c != '['){
		cout << "Error in FileParser::readIndex(): Expected '[', found '" << c << "'.\n";
		exit(1);
	}

	vector<int> indices;
	while(true){
		int i;
		bool foundBracket = readInt(&i, ']');
		indices.push_back(i);
		if(foundBracket)
			break;
	}

	Index *index = new Index(indices);
	return index;
}

void FileParser::readCoordinates(vector<double> *coordinates, int dimensions){
	char c;
	ssin >> c;
	if(c != '('){
		cout << "Error in FileParser::readCoordinates(): Expected '[', found '" << c << "'.\n";
		exit(1);
	}

	while(true){
		double d;
		bool foundBracket = readDouble(&d, ')');
		coordinates->push_back(d);
		if(foundBracket)
			break;
	}

	if(dimensions != (int)coordinates->size()){
		cout << "Error in FileParser::readCoordinates(): Expected " << dimensions << " coordinates, found " << coordinates->size() << ".\n";
		exit(1);
	}
}

void FileParser::readSpecifiers(vector<int> *specifiers, int numSpecifiers){
	char c;
	ssin >> c;
	if(c != '<'){
		cout << "Error in FileParser::readSpecifiers(): Expected '<', found '" << c << "'.\n";
		exit(1);
	}

	while(true){
		int i;
		bool foundBracket = readInt(&i, '>');
		specifiers->push_back(i);
		if(foundBracket)
			break;
	}

	if(numSpecifiers != (int)specifiers->size()){
		cout << "Error in FileParser::readSpecifiers(): Expected " << numSpecifiers << " specifiers, found " << specifiers->size() << ".\n";
		exit(1);
	}
}

bool FileParser::readComplex(complex<double> *c){
	double real;
	if(readDouble(&real))
		return 1;

	double imag;
	if(readDouble(&imag))
		return 1;

	*c = complex<double>(real, imag);

	return 0;
}

bool FileParser::readDouble(double *d, char endChar){
	string word;
	if(!(ssin >> word)){
		cout << "Error in FileParser::readDouble(): Reached end of file while trying to read floating point number.\n";
		exit(1);
	}

	//Pointer to first char after number. Used to indicate whether word is a number or not.
	char *p;

	*d = strtod(word.c_str(), &p);
	if(*p == endChar)
		return 1;
	if(*p){
		cout << "Error in FileParser::readDouble(): Expected floating point, found '" << word << "'.\n";
		exit(1);
	}

	return 0;
}

bool FileParser::readInt(int *i, char endChar){
	string word;
	if(!(ssin >> word)){
		cout << "Error in FileParser::readInt(): Reached end of file while trying to read integer.\n";
		exit(1);
	}

	//Pointer to first char after number. Used to indicate whether word is a number or not.
	char *p;

	*i = strtol(word.c_str(), &p, 10);
	if(*p == endChar)
		return 1;
	if(*p){
		cout << "Error in FileParser::readInt(): Expected integer, found '" << word << "'.\n";
		exit(1);
	}

	return 0;
}

};
