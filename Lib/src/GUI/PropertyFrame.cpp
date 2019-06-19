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

/** @file PropertyFrame.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Plot/Plotter.h"
#include "TBTK/DOSPanel.h"
#include "TBTK/LDOSPanel.h"
#include "TBTK/PropertyFrame.h"

using namespace std;

namespace TBTK{

PropertyFrame::PropertyFrame(
	wxWindow *parent
) :
	wxFrame(
		parent,
		wxID_ANY,
		"Property frame",
		wxDefaultPosition,
		wxSize(600, 400),
		wxDEFAULT_FRAME_STYLE,
		wxFrameNameStr
	)
{
	currentPanel = nullptr;
	currentPropertyType = PropertyType::None;

	SetSizer(new wxBoxSizer(wxHORIZONTAL));

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
}

PropertyFrame::~PropertyFrame(){
}

void PropertyFrame::setProperty(const Property::DOS &dos){
	setPropertyPanel(PropertyType::DOS);

	((DOSPanel*)currentPanel)->setDOS(dos);

	Refresh();
}

void PropertyFrame::setProperty(const Property::LDOS &ldos){
	setPropertyPanel(PropertyType::LDOS);

	((LDOSPanel*)currentPanel)->setLDOS(ldos);
	currentPropertyType = PropertyType::LDOS;

	Refresh();
}

BEGIN_EVENT_TABLE(PropertyFrame, wxFrame)

EVT_PAINT(PropertyFrame::onPaintEvent)
EVT_SIZE(PropertyFrame::onSizeEvent)
EVT_MENU(wxID_OPEN, PropertyFrame::onMenuFileOpen)
EVT_MENU(wxID_EXIT, PropertyFrame::onMenuFileQuit)
EVT_MENU(wxID_ABOUT, PropertyFrame::onMenuHelpAbout)

END_EVENT_TABLE()

void PropertyFrame::onPaintEvent(wxPaintEvent &event){
}

void PropertyFrame::onSizeEvent(wxSizeEvent &event){
	Layout();
	Refresh();
	event.Skip();
}

void PropertyFrame::onMenuFileOpen(wxCommandEvent &event){
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
			if(id.compare("DOS") == 0){
				Property::DOS dos(
					serialization,
					Serializable::Mode::JSON
				);
				setProperty(dos);
			}
			else if(id.compare("LDOS") == 0){
				Property::LDOS ldos(
					serialization,
					Serializable::Mode::JSON
				);
				setProperty(ldos);
			}
		}
	}
	openDialog->Destroy();
}

void PropertyFrame::onMenuFileQuit(wxCommandEvent &event){
	Close(false);
}

void PropertyFrame::onMenuHelpAbout(wxCommandEvent &event){
	wxLogMessage(
		_T(
			"TBTKPropertyViewer.\n"
			"\n"
			"View calculated properties.\n"
		)
	);
}

void PropertyFrame::setPropertyPanel(PropertyType propertyType){
	if(currentPropertyType == propertyType)
		return;

	switch(currentPropertyType){
	case PropertyType::None:
		break;
	case PropertyType::DOS:
		((DOSPanel*)currentPanel)->Destroy();
		break;
	case PropertyType::LDOS:
		((LDOSPanel*)currentPanel)->Destroy();
		break;
	default:
		TBTKExit(
			"PropertyFrame::~PropertyFrame()",
			"Unknown PropertyType.",
			"This should never happen, contact the developer."
		);
	}

	currentPropertyType = propertyType;

	switch(currentPropertyType){
	case PropertyType::None:
		break;
	case PropertyType::DOS:
		currentPanel = new DOSPanel(this);
		break;
	case PropertyType::LDOS:
		currentPanel = new LDOSPanel(this);
		break;
	default:
		TBTKExit(
			"PropertyFrame::~PropertyFrame()",
			"Unknown PropertyType.",
			"This should never happen, contact the developer."
		);
	}

	GetSizer()->Add(currentPanel, 1, wxEXPAND);
	Layout();
}

};	//End of namespace TBTK
