/** @file Streams.cpp
 *
 *  @author Kristofer Björnson
 */

#include "../include/Streams.h"

#include <iostream>

using namespace std;

namespace TBTK{
namespace Util{

Streams::DynamicOstream Streams::out(&cout);
Streams::DynamicOstream Streams::log(&cout);
Streams::DynamicOstream Streams::err(&cout);
//ostream &Streams::out = cout;
//ostream &Streams::log = cout;
//ostream &Streams::err = cout;
Streams::NullBuffer Streams::nullBuffer;
ostream Streams::null(&nullBuffer);

Streams::DynamicOstream::DynamicOstream(ostream *streamPointer){
	this->streamPointer = streamPointer;
}

Streams::DynamicOstream::~DynamicOstream(){
}

ostream& Streams::DynamicOstream::operator<<(bool val){
	return (*streamPointer << val);
}

ostream& Streams::DynamicOstream::operator<<(short val){
	return (*streamPointer << val);
}

ostream& Streams::DynamicOstream::operator<<(unsigned short val){
	return (*streamPointer << val);
}

ostream& Streams::DynamicOstream::operator<<(int val){
	return (*streamPointer << val);
}

ostream& Streams::DynamicOstream::operator<<(unsigned int val){
	return (*streamPointer << val);
}

ostream& Streams::DynamicOstream::operator<<(long val){
	return (*streamPointer << val);
}

ostream& Streams::DynamicOstream::operator<<(unsigned long val){
	return (*streamPointer << val);
}

ostream& Streams::DynamicOstream::operator<<(long long val){
	return (*streamPointer << val);
}

ostream& Streams::DynamicOstream::operator<<(unsigned long long val){
	return (*streamPointer << val);
}

ostream& Streams::DynamicOstream::operator<<(float val){
	return (*streamPointer << val);
}

ostream& Streams::DynamicOstream::operator<<(double val){
	return (*streamPointer << val);
}

ostream& Streams::DynamicOstream::operator<<(long double val){
	return (*streamPointer << val);
}

ostream& Streams::DynamicOstream::operator<<(void* val){
	return (*streamPointer << val);
}

ostream& Streams::DynamicOstream::operator<<(streambuf *sb){
	return (*streamPointer << sb);
}

ostream& Streams::DynamicOstream::operator<<(ostream& (*pf)(ostream&)){
	return (*streamPointer << pf);
}

ostream& Streams::DynamicOstream::operator<<(ios& (*pf)(ios&)){
	return (*streamPointer << pf);
}

ostream& Streams::DynamicOstream::operator<<(ios_base& (*pf)(ios_base&)){
	return (*streamPointer << pf);
}

ostream& Streams::DynamicOstream::operator<<(const char *chars){
	return (*streamPointer << chars);
}

int Streams::NullBuffer::overflow(int c){
	return c;
}

void Streams::muteOut(){
	out.setStream(&null);
}

void Streams::muteLog(){
	log.setStream(&null);
}

void Streams::muteErr(){
	err.setStream(&null);
}

};	//End of namespace Util
};	//End of namespace TBTK
