/* Copyright 2017 Kristofer Björnson
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

/** @file DataManagerFrame.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/DataManagerFrame.h"
#include "TBTK/ParameterSliderPanel.h"
#include "TBTK/PropertyFrame.h"
#include "TBTK/Streams.h"

#include <iomanip>

using namespace std;

namespace TBTK{

DataManagerFrame::DataManagerFrame(
	wxWindow *parent
) :
	wxFrame(
		parent,
		wxID_ANY,
		"DataManager frame",
		wxDefaultPosition,
		wxSize(200, 300),
		wxDEFAULT_FRAME_STYLE,
		wxFrameNameStr
	)
{
	SetSizer(new wxBoxSizer(wxVERTICAL));

	dataTypeSizer = new wxBoxSizer(wxHORIZONTAL);
	parameterSizer = new wxBoxSizer(wxVERTICAL);

	GetSizer()->Add(dataTypeSizer, 0, wxEXPAND);
	GetSizer()->Add(parameterSizer, 0, wxEXPAND);

	wxMenuBar *menuBar = new wxMenuBar();

	wxMenu *fileMenu = new wxMenu();
	fileMenu->Append(wxID_OPEN, _T("&Open"));
	fileMenu->AppendSeparator();
	fileMenu->Append(wxID_EXIT, _T("&Quit"));
	menuBar->Append(fileMenu, _T("&File"));

	wxMenu *helpMenu = new wxMenu();
	helpMenu->Append(wxID_ABOUT, _T("&About"));
	menuBar->Append(helpMenu, _T("Help"));

	SetMenuBar(menuBar);

	dataManager = nullptr;
	isOwner = false;
}

DataManagerFrame::~DataManagerFrame(){
}

void DataManagerFrame::setDataManager(const DataManager *dataManager, bool isOwner){
	if(this->dataManager != nullptr && this->isOwner)
		delete this->dataManager;

	for(unsigned int n = 0; n < checkboxes.size(); n++)
		checkboxes.at(n)->Destroy();
	checkboxes.clear();

	for(unsigned int n = 0; n < parameterSliders.size(); n++)
		parameterSliders.at(n)->Destroy();
	parameterSliders.clear();

	for(unsigned int n = 0; n < currentDataWindows.size(); n++)
		if(currentDataWindows.at(n) != nullptr)
			currentDataWindows.at(n)->Destroy();
	currentDataWindows.clear();

	this->dataManager = dataManager;
	this->isOwner = isOwner;

	for(unsigned int n = 0; n < dataManager->getNumDataTypes(); n++){
		wxCheckBox *checkbox = new wxCheckBox(
			this,
			wxID_ANY,
			dataManager->getDataType(n)
		);
		checkboxes.push_back(
			checkbox
		);
		dataTypeSizer->Add(checkbox, 1, 0);

		checkbox->Bind(
			wxEVT_CHECKBOX,
			&DataManagerFrame::onCheckboxEvent,
			this
		);

		currentDataWindows.push_back(nullptr);
	}

	for(unsigned int n = 0; n < dataManager->getNumParameters(); n++){
		ParameterSliderPanel *parameterSliderPanel = new ParameterSliderPanel(
			this,
			wxID_ANY,
			dataManager->getParameterName(n),
			dataManager->getLowerBound(n),
			dataManager->getUpperBound(n),
			dataManager->getNumTicks(n),
			0
		);
		parameterSizer->Add(parameterSliderPanel, 0, wxEXPAND);

		parameterSliders.push_back(parameterSliderPanel);

		parameterSliderPanel->Bind(
			wxEVT_SCROLL_CHANGED,
			&DataManagerFrame::onSliderChange,
			this
		);
	}

	Layout();
	Refresh();
}

BEGIN_EVENT_TABLE(DataManagerFrame, wxFrame)

EVT_PAINT(DataManagerFrame::onPaintEvent)
EVT_SIZE(DataManagerFrame::onSizeEvent)
EVT_MENU(wxID_OPEN, DataManagerFrame::onMenuFileOpen)
EVT_MENU(wxID_EXIT, DataManagerFrame::onMenuFileQuit)
EVT_MENU(wxID_ABOUT, DataManagerFrame::onMenuHelpAbout)

END_EVENT_TABLE()

void DataManagerFrame::onPaintEvent(wxPaintEvent &event){
}

void DataManagerFrame::onSizeEvent(wxSizeEvent &event){
	Layout();
	Refresh();
	event.Skip();
}

void DataManagerFrame::onMenuFileOpen(wxCommandEvent &event){
	wxFileDialog *openDialog = new wxFileDialog(
		this,
		_T("Choose file"),
		_(""),
		_(""),
		_("*.json"),
		wxFD_OPEN
	);

	if(openDialog->ShowModal() == wxID_OK){
		ifstream fin(openDialog->GetPath());
		string serialization(istream_iterator<char>(fin), {});
		if(
			Serializable::hasID(
				serialization,
				Serializable::Mode::JSON
			)
		){
			string id = Serializable::getID(
				serialization,
				Serializable::Mode::JSON
			);
			if(id.compare("DataManager") == 0){
				DataManager *dataManager = new DataManager(
					serialization,
					Serializable::Mode::JSON
				);
				setDataManager(dataManager, true);
			}
		}
	}
	openDialog->Destroy();
}

void DataManagerFrame::onMenuFileQuit(wxCommandEvent &event){
	Close(false);
}

void DataManagerFrame::onMenuHelpAbout(wxCommandEvent &event){
	wxLogMessage(
		_T(
			"TBTKDataManagerViewer.\n"
			"\n"
			"View data handled by a DataManager.\n"
		)
	);
}

void DataManagerFrame::onCheckboxEvent(wxCommandEvent &event){
	for(unsigned int n = 0; n < dataManager->getNumDataTypes(); n++){
		if(currentDataWindows.at(n) == nullptr && checkboxes.at(n)->GetValue()){
			PropertyFrame *propertyFrame = new PropertyFrame(this);
			propertyFrame->Show();
			currentDataWindows.at(n) = propertyFrame;
		}
		else if(
			currentDataWindows.at(n) != nullptr
			&& !checkboxes.at(n)->GetValue()
		){
			currentDataWindows.at(n)->Destroy();
			currentDataWindows.at(n) = nullptr;
		}
	}

	updateOpenWindows();
}

void DataManagerFrame::onSliderChange(wxScrollEvent &event){
	updateOpenWindows();
}

void DataManagerFrame::updateOpenWindows(){
	for(unsigned int n = 0; n < dataManager->getNumDataTypes(); n++){
		if(currentDataWindows.at(n) != nullptr){
			vector<unsigned int> dataPoint;
			for(
				unsigned int n = 0;
				n < parameterSliders.size();
				n++
			){
				dataPoint.push_back(
					parameterSliders.at(n)->getTick()
				);
			}
			unsigned int id = dataManager->getID(dataPoint);

			switch(dataManager->getFileType(dataManager->getDataType(n))){
			case DataManager::FileType::SerializableJSON:
			{
				string filename = dataManager->getFilename(
					dataManager->getDataType(n),
					id
				);

				ifstream fin(dataManager->getPath() + filename);
				if(!fin)
					break;

				string serialization(istream_iterator<char>(fin), {});
				string serializationID = Serializable::getID(
					serialization,
					Serializable::Mode::JSON
				);
				if(serializationID.compare("DOS") == 0){
					((PropertyFrame*)currentDataWindows.at(n))->setProperty(
						Property::DOS(
							serialization,
							Serializable::Mode::JSON
						)
					);
				}
				else if(serializationID.compare("LDOS") == 0){
					((PropertyFrame*)currentDataWindows.at(n))->setProperty(
						Property::LDOS(
							serialization,
							Serializable::Mode::JSON
						)
					);
				}

				break;
			}
			default:
				TBTKExit(
					"DataManagerFrame::DataManagerFrame()",
					"Only DataManager::FileType::SerializableJSON"
					<< " is supported yet.",
					""
				);
			}
		}
	}
}

};	//End of namespace TBTK
