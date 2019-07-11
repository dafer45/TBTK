/* Copyright 2016 Kristofer Björnson and Andreas Theiler
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/** @file FileParser.cpp
 *
 *  @author Kristofer Björnson
 *  @author Andreas Theiler
 */

#include "TBTK/FileParser.h"
#include "TBTK/Geometry.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"

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

void FileParser::writeParameterSet(
	const ParameterSet &parameterSet,
	string filename
){
	ofstream fout(filename);

	for(int n = 0; n < parameterSet.getNumInt(); n++){
		fout << "int\t" << parameterSet.getIntName(n) << "\t= "
			<< parameterSet.getIntValue(n) << "\n";
	}
	for(int n = 0; n < parameterSet.getNumDouble(); n++){
		fout << "double\t" << parameterSet.getDoubleName(n) << "\t= "
			<< parameterSet.getDoubleValue(n) << "\n";
	}
	for(int n = 0; n < parameterSet.getNumComplex(); n++){
		fout << "complex\t" << parameterSet.getComplexName(n) << "\t= "
			<< parameterSet.getComplexValue(n) << "\n";
	}
	for(int n = 0; n < parameterSet.getNumString(); n++){
		fout << "string\t" << parameterSet.getStringName(n) << "\t= "
			<< parameterSet.getStringValue(n) << "\n";
	}
	for(int n = 0; n < parameterSet.getNumBool(); n++){
		fout << "bool\t" << parameterSet.getBoolName(n) << "\t= "
			<< parameterSet.getBoolValue(n) << "\n";
	}

	fout.close();
}

ParameterSet FileParser::readParameterSet(string fileName){
	ParameterSet parameterSet;

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
			TBTKAssert(
				equalitySign == '=',
				"FileParser::readParameterSet()",
				"Expected '=' but found '" << equalitySign << "'.",
				""
			);
			int value;
			ssin >> value;
			parameterSet.addInt(name, value);
		}
		else if(type.compare("double") == 0){
			string name;
			ssin >> name;
			char equalitySign;
			ssin >> equalitySign;
			TBTKAssert(
				equalitySign == '=',
				"FileParser::readParameterSet()",
				"Expected '=' but found '" << equalitySign << "'.",
				""
			);
			double value;
			ssin >> value;
			parameterSet.addDouble(name, value);
		}
		else if(type.compare("complex") == 0){
			string name;
			ssin >> name;
			char equalitySign;
			ssin >> equalitySign;
			TBTKAssert(
				equalitySign == '=',
				"FileParser::readParameterSet()",
				"Expected '=' but found '" << equalitySign << "'.",
				""
			);
			complex<double> value;
			ssin >> value;
			parameterSet.addComplex(name, value);
		}
		else if(type.compare("string") == 0){
			string name;
			ssin >> name;
			char equalitySign;
			ssin >> equalitySign;
			TBTKAssert(
				equalitySign == '=',
				"FileParser::readParameterSet()",
				"Expected '=' but found '" << equalitySign << "'.",
				""
			);
			string value;
			getline(ssin, value);
			int first = value.find_first_not_of(" \t");
			int last = value.find_last_not_of(" \t");
			value = value.substr(first, last - first + 1);
			parameterSet.addString(name, value);
		}
		else if(type.compare("bool") == 0){
			string name;
			ssin >> name;
			char equalitySign;
			ssin >> equalitySign;
			TBTKAssert(
				equalitySign == '=',
				"FileParser::readParameterSet()",
				"Expected '=' but found '" << equalitySign << "'.",
				""
			);
			bool value;
			ssin >> value;
			parameterSet.addBool(name, value);
		}
		else{
			TBTKExit(
				"FileParser::readParametersSet()",
				"Expected type but found '" + type + "'.",
				""
			);
		}
/*		if(!getline(ssin, line)){
//			TBTKExit(
				"FileParser::readAmplitudes()",
				"Reached end of file while searching for 'Amplitudes:'.",
				""
			);
		}

		unsigned int pos = line.find();
		if(line.find("Amplitudes:") != string::npos){
			int mode = readParameter("Mode", "Amplitude");
			if(mode < 0 || mode > 1){
				TBTKExit(
					"FileParser::readAmplitudes()",
					"Only Amplitude mode 0 and 1 supported.",
					""
				);
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
	TBTKAssert(
		fin,
		"FileParser::readInput()",
		"Unable to open '" + fileName + "'.",
		""
	);

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
	for(unsigned int n = 0; n < index.getSize(); n++){
		if(n != 0)
			ss << " ";
		ss << index.at(n);
	}
	ss << "]";

	fout << ss.str();
}

void FileParser::writeCoordinates(
//	const double *coordinates,
	const vector<double> &coordinates/*,
	int numCoordinates*/
){
	stringstream ss;
	ss << "(";
	for(unsigned int n = 0; n < coordinates.size(); n++){
		if(n != 0)
			ss << " ";
		ss << coordinates[n];
	}
	ss << ")";

	fout << ss.str();
}

//void FileParser::writeSpecifiers(const int *specifiers, int numSpecifiers){
void FileParser::writeSpecifiers(const vector<int> &specifiers){
	stringstream ss;
	ss << "<";
	for(unsigned int n = 0; n < specifiers.size(); n++){
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

	for(
		HoppingAmplitudeSet::ConstIterator iterator
			= model->getHoppingAmplitudeSet().cbegin();
		iterator != model->getHoppingAmplitudeSet().cend();
		++iterator
	){
		switch(amplitudeMode){
		case AmplitudeMode::ALL:
			fout << left << setw(30);
			write((*iterator).getAmplitude());
			fout << left << setw(20);
			write((*iterator).getToIndex());
			write((*iterator).getFromIndex());
			writeLineBreaks(1);
			break;
		case AmplitudeMode::ALL_EXCEPT_HC:
		{
			int from = model->getBasisIndex((*iterator).getFromIndex());
			int to = model->getBasisIndex((*iterator).getToIndex());
			if(from <= to){
				fout << left << setw(30);
				write((*iterator).getAmplitude());
				fout << setw(20);
				write((*iterator).getToIndex());
				write((*iterator).getFromIndex());
				writeLineBreaks(1);
			}
			break;
		}
		default:
			TBTKExit(
				"FileParser::writeAmplitudes()",
				"Unsupported amplitudeMode (" << static_cast<int>(amplitudeMode) << ").",
				""
			);
		}
	}
}

void FileParser::writeGeometry(Model *model){
//	Geometry *geometry = model->getGeometry();
	Geometry &geometry = model->getGeometry();

/*	if(geometry == NULL){
		fout << "Geometry: None\n";
		return;
	}
	else{
		fout << "Geometry:\n";
	}*/
	fout << "Geometry:\n";

	int dimensions = geometry.getDimensions();
//	const int numSpecifiers = geometry->getNumSpecifiers();
	fout << left << setw(30) << "Dimensions" << "= " << dimensions << "\n";
//	fout << left << setw(30) << "Num specifiers" << "= " << numSpecifiers << "\n";

	Index dummyIndex({-1});
	Index &prevIndex = dummyIndex;//Start with dummy index
	for(
		HoppingAmplitudeSet::ConstIterator iterator
			= model->getHoppingAmplitudeSet().cbegin();
		iterator != model->getHoppingAmplitudeSet().cend();
		++iterator
	){
		const Index &index = (*iterator).getFromIndex();
		if(!index.equals(prevIndex)){
//			const double *coordinates = geometry->getCoordinates(index);
			const vector<double> &coordinates = geometry.getCoordinate(index);
//			const int *specifiers = geometry->getSpecifiers(index);
			fout << left << setw(30);
//			writeCoordinates(coordinates, dimensions);
			writeCoordinates(coordinates);
			fout << setw(20);
//			writeSpecifiers(specifiers, numSpecifiers);
			writeSpecifiers({});
			write(index);
			writeLineBreaks(1);

			prevIndex = index;
		}
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
			TBTKExit(
				"FileParser::removeComments()",
				"Unknown state.",
				""
			);
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
			TBTKExit(
				"FileParser::removeInitialWhiteSpaces()",
				"Unknown state.",
				""
			);
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
		TBTKAssert(
			getline(ssin, line),
			"FileParser::readAmplitudes()",
			"Reached end of file while searching for 'Amplitudes:'.",
			""
		);

		if(line.find("Amplitudes:") != string::npos){
			int mode = readParameter("Mode", "Amplitude");
			if(mode < 0 || mode > 1){
				TBTKExit(
					"FileParser::readAmplitudes()",
					"Only Amplitude mode 0 and 1 supported.",
					""
				);
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
			*model << *ha;
			break;
		case AmplitudeMode::ALL_EXCEPT_HC:
		{
/*			const Index &from = ha->fromIndex;
			const Index &to = ha->toIndex;*/
			const Index &from = ha->getFromIndex();
			const Index &to = ha->getToIndex();
			if(from.equals(to))
				*model << *ha;
			else
				*model << *ha + HC;
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
			Streams::log << "Warning in FileParser::readAmplitudes(): Reached end of file while searching for 'Geometry:'.\n";
			Streams::log << "\tNo Geometry loaded.\n";
			Streams::log << "\tAdd 'Geometry: None' after amplitude list to disable warning.\n";
			return;
		}

		if(line.find("Geometry:") != string::npos){
			if(line.find("None") != string::npos)
				return;

			dimensions = readParameter("Dimensions", "Geometry");
			numSpecifiers = readParameter("Num specifiers", "Geometry");

			break;
		}
	}

//	model->createGeometry(dimensions, numSpecifiers);
//	Geometry *geometry = model->getGeometry();
	Geometry &geometry = model->getGeometry();

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

//		geometry->setCoordinates(*index, coordinates, specifiers);
		geometry.setCoordinate(*index, coordinates);

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

	TBTKAssert(
		getline(ssin, line),
		"FileParser::readParameter()",
		"Expected parameter '" << parameterName << "' for structure '" << parentStructure << "'.",
		""
	);

	size_t position = line.find(parameterName);
	TBTKAssert(
		position != string::npos,
		"FileParser::readAmplitudes()",
		"Expected parameter '" << parameterName << "' for structure '" << parentStructure << "'.",
		""
	);
	line = line.substr(position+4);

	position = line.find("=");
	TBTKAssert(
		position != string::npos,
		"FileParser::readAmplitudes()",
		"Expected '=' after " << parameterName << ".",
		""
	);
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
	TBTKAssert(
		to != NULL,
		"FileParser::readHoppingAmplitude()",
		"Faile to read index.",
		""
	);

	Index *from = readIndex();
	TBTKAssert(
		to != NULL,
		"FileParser::readHoppingAmplitude()",
		"Faile to read index.",
		""
	);

	HoppingAmplitude *ha = new HoppingAmplitude(amplitude, *to, *from);
	delete to;
	delete from;

	return ha;
}

Index* FileParser::readIndex(){
	char c;
	ssin >> c;
	TBTKAssert(
		c == '[',
		"FileParser::readIndex()",
		"Expected '[', found '" << c << "'.",
		""
	);

	vector<Subindex> indices;
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
	TBTKAssert(
		c == '(',
		"FileParser::readCoordinates()",
		"Expected '[', found '" << c << "'.",
		""
	);

	while(true){
		double d;
		bool foundBracket = readDouble(&d, ')');
		coordinates->push_back(d);
		if(foundBracket)
			break;
	}

	TBTKAssert(
		dimensions == (int)coordinates->size(),
		"FileParser::readCoordinates()",
		"Expected " << dimensions << " coordinates, found " << coordinates->size() << ".",
		""
	);
}

void FileParser::readSpecifiers(vector<int> *specifiers, int numSpecifiers){
	char c;
	ssin >> c;
	TBTKAssert(
		c == '<',
		"FileParser::readSpecifiers()",
		"Expected '<', found '" << c << "'.",
		""
	);

	while(true){
		int i;
		bool foundBracket = readInt(&i, '>');
		specifiers->push_back(i);
		if(foundBracket)
			break;
	}

	TBTKAssert(
		numSpecifiers == (int)specifiers->size(),
		"FileParser::readSpecifiers()",
		"Expected " << numSpecifiers << " specifiers, found " << specifiers->size() << ".",
		""
	);
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
	TBTKAssert(
		(ssin >> word),
		"FileParser::readDouble()",
		"Reached end of file while trying to read floating point number.",
		""
	);

	//Pointer to first char after number. Used to indicate whether word is a number or not.
	char *p;

	*d = strtod(word.c_str(), &p);
	if(*p == endChar)
		return 1;
	TBTKAssert(
		*p == 0,
		"FileParser::readDouble()",
		"Expected floating point, found '" << word << "'.",
		""
	);

	return 0;
}

bool FileParser::readInt(int *i, char endChar){
	string word;
	TBTKAssert(
		(ssin >> word),
		"FileParser::readInt()",
		"Reached end of file while trying to read integer.",
		""
	);

	//Pointer to first char after number. Used to indicate whether word is a number or not.
	char *p;

	*i = strtol(word.c_str(), &p, 10);
	if(*p == endChar)
		return 1;
	TBTKAssert(
		*p == 0,
		"FileParser::readInt()",
		"Expected integer, found '" << word << "'.",
		""
	);

	return 0;
}

};
