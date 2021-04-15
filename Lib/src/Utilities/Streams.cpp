/* Copyright 2016 Kristofer Björnson
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

/** @file Streams.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"

#include <fstream>
#include <iostream>

using namespace std;

namespace TBTK{

//ostream Streams::out(new ForkBuffer(&cout, &Streams::log));
//stream Streams::log(&nullBuffer);
//ostream Streams::err(new ForkBuffer(&cout, &Streams::log));
ostream Streams::out(&stdOutBuffer);
//ostream Streams::log(stdLogBuffer.rdbuf());
ostream Streams::log(&Streams::stdLogBuffer);
ostream Streams::err(&stdErrBuffer);
ostream Streams::null(&nullBuffer);
Streams::NullBuffer Streams::nullBuffer;

//Line one and three below are not memory leak even if they appear as such. The
//created objects are intended to last throughout the existence of the program.
Streams::ForkBuffer Streams::stdOutBuffer(&cout, &Streams::log);
Streams::LogBuffer Streams::stdLogBuffer;
Streams::ForkBuffer Streams::stdErrBuffer(&cerr, &Streams::log);
ofstream Streams::logFile;

int Streams::NullBuffer::overflow(int c){
	return c;
}

void Streams::setStdMuteOut(){
	stdOutBuffer.mute(0, true);
	stdOutBuffer.mute(1, false);
}

void Streams::setStdMuteErr(){
	stdErrBuffer.mute(0, true);
	stdErrBuffer.mute(1, false);
}

void Streams::openLog(std::string fileName){
	stdLogBuffer.open(fileName);

/*	Streams::log << TBTK_ABOUT << "\n";
	Streams::log << "Date: " << TBTK_GET_CURRENT_TIME_STRING() << "\n";
	Streams::log << "\n";*/
	Streams::log << TBTK_RUNTIME_CONTEXT_STRING << "\n";
}

void Streams::closeLog(){
	Streams::log << "\n";
	Streams::log << "Date: " << TBTK_GET_CURRENT_TIME_STRING();
	stdLogBuffer.close();
}

bool Streams::logIsOpen(){
	return stdLogBuffer.isOpen();
}

Streams::ForkBuffer::ForkBuffer(
	basic_ostream<char,
	char_traits<char>> *ostream1,
	basic_ostream<char,
	char_traits<char>> *ostream2
){
	ostreams[0] = ostream1;
	ostreams[1] = ostream2;

	isMute[0] = false;
	isMute[1] = false;
}

void Streams::ForkBuffer::mute(int n, bool isMute){
	this->isMute[n] = isMute;
}

int Streams::ForkBuffer::overflow(int c){
	for(int n = 0; n < 2; n++)
		if(ostreams[n] && !isMute[n])
			*(ostreams[n]) << (char)c;

	return c;
}

Streams::LogBuffer::LogBuffer(){
	fout.rdbuf()->pubsetbuf(0, 0);
}

Streams::LogBuffer::~LogBuffer(){
}

void Streams::LogBuffer::open(std::string fileName){
	if(fout.is_open()){
		//Do not use TBTKExit or TBTKAssert here. These rely on Streams
		//for output. cerr is used to ensure proper error messages also
		//in the case that Streams fail.
		cerr << "Error in Streams::LogBuffer::openFile(): Log file already open." << endl;
		exit(1);
	}

	fout.open(fileName);
};

void Streams::LogBuffer::close(){
	if(fout.is_open()){
		fout << flush;
		fout.close();
	}
	else{
		//Do not use TBTKExit or TBTKAssert here. These rely on Streams
		//for output. cerr is used to ensure proper error messages also
		//in the case that Streams fail.
		Streams::err << "Error in Streams::LogBuffer::closeFile(): No log file is open.\n";
		exit(1);
	}
}

bool Streams::LogBuffer::isOpen(){
	return fout.is_open();
}

int Streams::LogBuffer::overflow(int c){
	if(fout.is_open())
		fout << (char)c;
	return c;
}

};	//End of namespace TBTK
