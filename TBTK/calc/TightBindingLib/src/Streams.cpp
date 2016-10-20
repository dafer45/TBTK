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

/*Streams::DefaultOutStream::DefaultOutStream(){
}

ostream& Streams::DefaultOutStream::operator<<(bool val){
	Streams::log << val;
	return (cout << val);
}

ostream& Streams::DefaultOutStream::operator<<(short val){
	Streams::log << val;
	return (cout << val);
}

ostream& Streams::DefaultOutStream::operator<<(unsigned short val){
	Streams::log << val;
	return (cout << val);
}

ostream& Streams::DefaultOutStream::operator<<(int val){
	Streams::log << val;
	return (cout << val);
}

ostream& Streams::DefaultOutStream::operator<<(unsigned int val){
	Streams::log << val;
	return (cout << val);
}

ostream& Streams::DefaultOutStream::operator<<(long val){
	Streams::log << val;
	return (cout << val);
}

ostream& Streams::DefaultOutStream::operator<<(unsigned long val){
	Streams::log << val;
	return (cout << val);
}

ostream& Streams::DefaultOutStream::operator<<(long long val){
	Streams::log << val;
	return (cout << val);
}

ostream& Streams::DefaultOutStream::operator<<(unsigned long long val){
	Streams::log << val;
	return (cout << val);
}

ostream& Streams::DefaultOutStream::operator<<(float val){
	Streams::log << val;
	return (cout << val);
}

ostream& Streams::DefaultOutStream::operator<<(double val){
	Streams::log << val;
	return (cout << val);
}

ostream& Streams::DefaultOutStream::operator<<(long double val){
	Streams::log << val;
	return (cout << val);
}

ostream& Streams::DefaultOutStream::operator<<(void* val){
	Streams::log << val;
	return (cout << val);
}

ostream& Streams::DefaultOutStream::operator<<(streambuf *sb){
	Streams::log << sb;
	return (cout << sb);
}

ostream& Streams::DefaultOutStream::operator<<(ostream& (*pf)(ostream&)){
	Streams::log << pf;
	return (cout << pf);
}

ostream& Streams::DefaultOutStream::operator<<(ios& (*pf)(ios&)){
	Streams::log << pf;
	return (cout << pf);
}

ostream& Streams::DefaultOutStream::operator<<(ios_base& (*pf)(ios_base&)){
	Streams::log << pf;
	return (cout << pf);
}

ostream& Streams::DefaultOutStream::operator<<(const char *chars){
	Streams::log << chars;
	return (cout << chars);
}

Streams::DefaultLogStream::DefaultLogStream(string filename) : fstream(filename){
}

ostream& Streams::DefaultLogStream::operator<<(bool val){
	return fstream::operator<<(val);
}

ostream& Streams::DefaultLogStream::operator<<(short val){
	return fstream::operator<<(val);
}

ostream& Streams::DefaultLogStream::operator<<(unsigned short val){
	return fstream::operator<<(val);
}

ostream& Streams::DefaultLogStream::operator<<(int val){
	return fstream::operator<<(val);
}

ostream& Streams::DefaultLogStream::operator<<(unsigned int val){
	return fstream::operator<<(val);
}

ostream& Streams::DefaultLogStream::operator<<(long val){
	return fstream::operator<<(val);
}

ostream& Streams::DefaultLogStream::operator<<(unsigned long val){
	return fstream::operator<<(val);
}

ostream& Streams::DefaultLogStream::operator<<(long long val){
	return fstream::operator<<(val);
}

ostream& Streams::DefaultLogStream::operator<<(unsigned long long val){
	return fstream::operator<<(val);
}

ostream& Streams::DefaultLogStream::operator<<(float val){
	return fstream::operator<<(val);
}

ostream& Streams::DefaultLogStream::operator<<(double val){
	return fstream::operator<<(val);
}

ostream& Streams::DefaultLogStream::operator<<(long double val){
	return fstream::operator<<(val);
}

ostream& Streams::DefaultLogStream::operator<<(void* val){
	return fstream::operator<<(val);
}

ostream& Streams::DefaultLogStream::operator<<(streambuf *sb){
	return fstream::operator<<(sb);
}

ostream& Streams::DefaultLogStream::operator<<(ostream& (*pf)(ostream&)){
	return fstream::operator<<(pf);
}

ostream& Streams::DefaultLogStream::operator<<(ios& (*pf)(ios&)){
	return fstream::operator<<(pf);
}

ostream& Streams::DefaultLogStream::operator<<(ios_base& (*pf)(ios_base&)){
	return fstream::operator<<(pf);
}

ostream& Streams::DefaultLogStream::operator<<(const char *chars){
	return fstream::operator<<(chars);
}

Streams::DefaultErrStream::DefaultErrStream(){
}

ostream& Streams::DefaultErrStream::operator<<(bool val){
	Streams::log << val;
	return (cout << val);
}

ostream& Streams::DefaultErrStream::operator<<(short val){
	Streams::log << val;
	return (cout << val);
}

ostream& Streams::DefaultErrStream::operator<<(unsigned short val){
	Streams::log << val;
	return (cout << val);
}

ostream& Streams::DefaultErrStream::operator<<(int val){
	Streams::log << val;
	return (cout << val);
}

ostream& Streams::DefaultErrStream::operator<<(unsigned int val){
	Streams::log << val;
	return (cout << val);
}

ostream& Streams::DefaultErrStream::operator<<(long val){
	Streams::log << val;
	return (cout << val);
}

ostream& Streams::DefaultErrStream::operator<<(unsigned long val){
	Streams::log << val;
	return (cout << val);
}

ostream& Streams::DefaultErrStream::operator<<(long long val){
	Streams::log << val;
	return (cout << val);
}

ostream& Streams::DefaultErrStream::operator<<(unsigned long long val){
	Streams::log << val;
	return (cout << val);
}

ostream& Streams::DefaultErrStream::operator<<(float val){
	Streams::log << val;
	return (cout << val);
}

ostream& Streams::DefaultErrStream::operator<<(double val){
	Streams::log << val;
	return (cout << val);
}

ostream& Streams::DefaultErrStream::operator<<(long double val){
	Streams::log << val;
	return (cout << val);
}

ostream& Streams::DefaultErrStream::operator<<(void* val){
	Streams::log << val;
	return (cout << val);
}

ostream& Streams::DefaultErrStream::operator<<(streambuf *sb){
	Streams::log << sb;
	return (cout << sb);
}

ostream& Streams::DefaultErrStream::operator<<(ostream& (*pf)(ostream&)){
	Streams::log << pf;
	return (cout << pf);
}

ostream& Streams::DefaultErrStream::operator<<(ios& (*pf)(ios&)){
	Streams::log << pf;
	return (cout << pf);
}

ostream& Streams::DefaultErrStream::operator<<(ios_base& (*pf)(ios_base&)){
	Streams::log << pf;
	return (cout << pf);
}

ostream& Streams::DefaultErrStream::operator<<(const char *chars){
	Streams::log << chars;
	return (cout << chars);
}*/

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
