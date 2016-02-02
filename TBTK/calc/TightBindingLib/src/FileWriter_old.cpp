#include "../include/FileWriter.h"
#include "../include/FileIO.h"
#include <string>
#include <sstream>
#include <iostream>

using namespace std;

void FileWriter::writeEV(double *ev, int size, string filename, string path){
	stringstream ss;
	ss << path;
	if(path.back() != '/')
		ss << "/";
	ss << filename;
	ofstream fout;
	fout.open(ss.str().c_str(), ios::out|ios::binary);

	writeHeader(fout, TB_FILE_TYPE_EV);

	writeInt(fout, size);
	for(int n = 0; n < size; n++)
		writeDouble(fout, ev[n]);

	fout.close();
}

void FileWriter::writeDOS(double *dos, double u_lim, double l_lim, int resolution, string filename, string path){
	stringstream ss;
	ss << path;
	if(path.back() != '/')
		ss << "/";
	ss << filename;
	ofstream fout;
	fout.open(ss.str().c_str(), ios::out|ios::binary);

	writeHeader(fout, TB_FILE_TYPE_DOS);

	writeDouble(fout, u_lim);
	writeDouble(fout, l_lim);
	writeInt(fout, resolution);
	for(int n = 0; n < resolution; n++)
		writeDouble(fout, dos[n]);

	fout.close();
}

void FileWriter::writeHeader(ofstream &fout, TBFileType tbFileType){
	TBFileHeader header;
	header.magic[0] = 'M';
	header.magic[1] = 'A';
	header.magic[2] = 'G';
	header.magic[3] = 'I';

	for(int n = 0; n < 4; n++)
		fout << header.magic[n];

	header.file_version = 0;
	header.software_version = 1;
	header.checksum = -1000000000;
	header.fileType = tbFileType;

	writeInt(fout, header.file_version);
	writeInt(fout, header.software_version);
	writeInt(fout, header.checksum);
	writeInt(fout, header.fileType);
}

void FileWriter::writeShort(ofstream &fout, short s){
	fout << (unsigned char)((s & 0xFF00) >> 8);
	fout << (unsigned char)(s & 0x00FF);
}

void FileWriter::writeInt(ofstream &fout, int i){
	fout << (unsigned char)((i & 0xFF000000) >> 24);
	fout << (unsigned char)((i & 0x00FF0000) >> 16);
	fout << (unsigned char)((i & 0x0000FF00) >> 8);
	fout << (unsigned char)(i & 0x000000FF);
}

void FileWriter::writeFloat(ofstream &fout, float f){
	fout << (unsigned char)(((unsigned int)f & 0xFF000000) >> 24);
	fout << (unsigned char)(((unsigned int)f & 0x00FF0000) >> 16);
	fout << (unsigned char)(((unsigned int)f & 0x0000FF00) >> 8);
	fout << (unsigned char)((unsigned int)f & 0x000000FF);
}

void FileWriter::writeDouble(ofstream &fout, double d){
	fout.write((char*)&d, 8);
}
