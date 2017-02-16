#include "../include/MainApplication.h"

#include <OgreException.h>
#include <OgreConfigFile.h>
#include <OgreTextureManager.h>
#include <OgreEntity.h>
#include <OgreSceneNode.h>
#include <OgreLogManager.h>
#include <OgreQuaternion.h>
#include <OgreSubEntity.h>
#include <OgreTechnique.h>
#include <OgreHardwarePixelBuffer.h>
#include <OgreRenderTexture.h>

#include "FileReader.h"
#include "Model.h"
#include "Geometry.h"
#include "HoppingAmplitude.h"

#include <sstream>

#define TO_STRING2(x) #x
#define TO_STRING(x) TO_STRING2(x)

using namespace std;

MainApplication::MainApplication() :
	mRoot(NULL),
	mResourcesCfg(Ogre::BLANKSTRING),
	mPluginsCfg(Ogre::BLANKSTRING),
	mRotate(.13),
	mMove(25),
	cameraNode(NULL),
	cameraDirection(Ogre::Vector3::ZERO),
	geometryNode(NULL),
	mShutDown(0),
	screenshotCounter(0)
{
}

MainApplication::~MainApplication(){
	Ogre::WindowEventUtilities::removeWindowEventListener(mWindow, this);
	windowClosed(mWindow);
	if(mRoot != NULL)
		delete mRoot;
}

bool MainApplication::init(){
	//Resource files
	#ifdef _DEBUG
		mResourcesCfg = TO_STRING(EXECUTABLE_DIR) "/resources_d.cfg";
		mPluginsCfg = TO_STRING(EXECUTABLE_DIR) "/plugins_d.cfg";
		mConfigCfg = TO_STRING(EXECUTABLE_DIR) "ogre_d.cfg";
		mLog = TO_STRING(EXECUTABLE_DIR) "log_d.cfg";
	#else
		mResourcesCfg = TO_STRING(EXECUTABLE_DIR)  "/resources.cfg";
		mPluginsCfg = TO_STRING(EXECUTABLE_DIR)  "/plugins.cfg";
		mConfigCfg = TO_STRING(EXECUTABLE_DIR) "ogre.cfg";
		mLog = TO_STRING(EXECUTABLE_DIR) "log.cfg";
	#endif

	mRoot = new Ogre::Root(mPluginsCfg, mConfigCfg, mLog);

	//Load resource locations
	Ogre::ConfigFile cf;
	cf.load(mResourcesCfg);
	Ogre::String name, locType;
	Ogre::ConfigFile::SectionIterator secIt = cf.getSectionIterator();
	while(secIt.hasMoreElements()){
		Ogre::ConfigFile::SettingsMultiMap *settings = secIt.getNext();
		Ogre::ConfigFile::SettingsMultiMap::iterator it;
		for(it = settings->begin(); it != settings->end(); it++){
			locType = it->first;
			name = it->second;
			Ogre::ResourceGroupManager::getSingleton().addResourceLocation(name, locType);
		}
	}

	//Configure render system
	if(!(mRoot->restoreConfig() || mRoot->showConfigDialog()))
		return false;

	//Create render window
	mWindow = mRoot->initialise(true, "TBTK Visualizer");

	//Initialize resources
	Ogre::TextureManager::getSingleton().setDefaultNumMipmaps(5);
	Ogre::ResourceGroupManager::getSingleton().initialiseAllResourceGroups();

	//Create scene manager
	mSceneManager = mRoot->createSceneManager(Ogre::ST_GENERIC);

	//Initialize OIS
	Ogre::LogManager::getSingletonPtr()->logMessage("*** Initializing OIS ***");
	OIS::ParamList pl;
	size_t windowHnd = 0;
	std::ostringstream windowHndStr;

	mWindow->getCustomAttribute("WINDOW", &windowHnd);
	windowHndStr << windowHnd;
	pl.insert(std::make_pair(std::string("WINDOW"), windowHndStr.str()));

	mInputManager = OIS::InputManager::createInputSystem(pl);

	mKeyboard = static_cast<OIS::Keyboard*>(mInputManager->createInputObject(OIS::OISKeyboard, true));
	mMouse = static_cast<OIS::Mouse*>(mInputManager->createInputObject(OIS::OISMouse, true));

	//Set initial mouse clipping size and register event listener
	windowResized(mWindow);
	Ogre::WindowEventUtilities::addWindowEventListener(mWindow, this);

	//Register keyboard and mouse listeners
	mKeyboard->setEventCallback(this);
	mMouse->setEventCallback(this);

	return true;
}

bool MainApplication::go(){
	if(!init())
		return false;

	createScene();

	loadGeometry();

	//Register frame listener
	mRoot->addFrameListener(this);
	mRoot->startRendering();

	return true;
}

void MainApplication::createScene(){
	//Create camera
	mCamera = mSceneManager->createCamera("MainCamera");
	cameraNode = mSceneManager->getRootSceneNode()->createChildSceneNode("CameraNode");
	cameraNode->attachObject(mCamera);
	mCamera->setNearClipDistance(5);

	//Create viewport
	mViewport = mWindow->addViewport(mCamera);
	mViewport->setBackgroundColour(Ogre::ColourValue(0, 0, 0));
	mCamera->setAspectRatio(
		Ogre::Real(mViewport->getActualWidth())/Ogre::Real(mViewport->getActualHeight())
	);

	mSceneManager->setAmbientLight(Ogre::ColourValue(.5, .5, .5));
	light = mSceneManager->createLight();
	light->setPosition(20, 80, 50);
}

void MainApplication::deleteGeometry(){
	destroyAllAttachedMovableObjects(geometryNode);
	geometryNode->removeAndDestroyAllChildren();
	mSceneManager->destroySceneNode(geometryNode);
}

void MainApplication::destroyAllAttachedMovableObjects(Ogre::SceneNode *node){
	Ogre::SceneNode::ObjectIterator itObject = node->getAttachedObjectIterator();
	while(itObject.hasMoreElements()){
		Ogre::MovableObject *object = static_cast<Ogre::MovableObject*>(itObject.getNext());
		geometryNode->getCreator()->destroyMovableObject(object);
	}

	Ogre::SceneNode::ChildNodeIterator itChild = node->getChildIterator();
	while(itChild.hasMoreElements()){
		Ogre::SceneNode *childNode = static_cast<Ogre::SceneNode*>(itChild.getNext());
		destroyAllAttachedMovableObjects(childNode);
	}
}

void MainApplication::loadGeometry(){
	geometryNode = mSceneManager->getRootSceneNode()->createChildSceneNode("GeometryNode");

	TBTK::FileReader::setFileName(fileName);
	TBTK::Model *model = TBTK::FileReader::readModel();
	int dimensions = model->getGeometry()->getDimensions();
//	double atomRadius = 0.038;
	double atomRadius = 0.01;

	TBTK::HoppingAmplitudeSet::Iterator it = model->getHoppingAmplitudeSet()->getIterator();
	const TBTK::HoppingAmplitude *ha;
	int counter = 0;
	while((ha = it.getHA())){
		std::stringstream ss;
		ss << "Site" << counter++;
		Ogre::Entity *entity = mSceneManager->createEntity(ss.str(), Ogre::SceneManager::PT_SPHERE);
		Ogre::SceneNode *node = geometryNode->createChildSceneNode();
		node->attachObject(entity);

		const double *coordinates = model->getGeometry()->getCoordinates(ha->fromIndex);
		switch(dimensions){
		case 1:
			node->setPosition(0, 0, coordinates[0]);
			break;
		case 2:
			node->setPosition(coordinates[1], 0, coordinates[0]);
			break;
		case 3:
			node->setPosition(coordinates[1], coordinates[2], coordinates[0]);
			break;
		default:
			cout << "Error in MainApplication::loadGeometry(): Geometry must have dimension 1-3, but it is " << dimensions << "\n.";
			exit(1);
		}

		node->setScale(atomRadius, atomRadius, atomRadius);

		it.searchNextHA();
	}

	delete model;
}

void MainApplication::windowResized(Ogre::RenderWindow *rw){
	unsigned int width, height, depth;
	int left, top;
	rw->getMetrics(width, height, depth, left, top);
	const OIS::MouseState &ms = mMouse->getMouseState();
	ms.width = width;
	ms.height = height;
}

void MainApplication::windowClosed(Ogre::RenderWindow *rw){
	if(rw == mWindow){
		if(mInputManager){
			mInputManager->destroyInputObject(mMouse);
			mInputManager->destroyInputObject(mKeyboard);

			OIS::InputManager::destroyInputSystem(mInputManager);
			mInputManager = 0;
		}
	}
}

bool MainApplication::frameRenderingQueued(const Ogre::FrameEvent &event){
	cameraNode->translate(cameraDirection*event.timeSinceLastFrame, Ogre::Node::TS_LOCAL);

	if(mWindow->isClosed())
		return false;

	if(mShutDown)
		return false;

	mKeyboard->capture();
	mMouse->capture();

	return true;
}

bool MainApplication::keyPressed(const OIS::KeyEvent &ke){
	switch(ke.key){
	case OIS::KC_ESCAPE:
		mShutDown = true;
		break;
	case OIS::KC_UP:
	case OIS::KC_W:
		cameraDirection.z = -mMove;
		break;
	case OIS::KC_DOWN:
	case OIS::KC_S:
		cameraDirection.z = mMove;
		break;
	case OIS::KC_LEFT:
	case OIS::KC_A:
		cameraDirection.x = -mMove;
		break;
	case OIS::KC_RIGHT:
	case OIS::KC_D:
		cameraDirection.x = mMove;
		break;
	case OIS::KC_PGDOWN:
	case OIS::KC_E:
		cameraDirection.y = -mMove;
		break;
	case OIS::KC_PGUP:
	case OIS::KC_Q:
		cameraDirection.y = mMove;
		break;
	case OIS::KC_1:
		if(fileName != "TBTKResults_1.h5"){
			setFileName("TBTKResults_1.h5");
			deleteGeometry();
			loadGeometry();
		}
		break;
	case OIS::KC_2:
		if(fileName != "TBTKResults_2.h5"){
			setFileName("TBTKResults_2.h5");
			deleteGeometry();
			loadGeometry();
		}
		break;
	case OIS::KC_3:
		if(fileName != "TBTKResults_3.h5"){
			setFileName("TBTKResults_3.h5");
			deleteGeometry();
			loadGeometry();
		}
		break;
	case OIS::KC_4:
		if(fileName != "TBTKResults_4.h5"){
			setFileName("TBTKResults_4.h5");
			deleteGeometry();
			loadGeometry();
		}
		break;
	case OIS::KC_P:
	{
		stringstream ss;
		ss << "MainRenderTarget" << screenshotCounter;
		cout << "\n\n\n" << ss.str() << "\n\n\n";
		Ogre::TexturePtr tex = Ogre::TextureManager::getSingleton().createManual(
			ss.str(),
			Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
			Ogre::TextureType::TEX_TYPE_2D,
			1366,
			768,
			10000,
			0,
			Ogre::PixelFormat::PF_R8G8B8,
			Ogre::TextureUsage::TU_RENDERTARGET
		);

		Ogre::RenderTexture *renderTexture = tex->getBuffer()->getRenderTarget();

		renderTexture->addViewport(mCamera);
		int count = renderTexture->getNumViewports();

		renderTexture->getViewport(0)->setClearEveryFrame(true);
		renderTexture->getViewport(0)->setBackgroundColour(Ogre::ColourValue::Black);
		renderTexture->getViewport(0)->setOverlaysEnabled(false);

		renderTexture->update();

		ss.str("");
		ss << "./Screenshot" << screenshotCounter++ << ".png";
		cout << ss.str() << "\n";
		renderTexture->writeContentsToFile(ss.str());
		break;
	}
	default:
		break;
	}

	return true;
}

bool MainApplication::keyReleased(const OIS::KeyEvent &ke){
	switch(ke.key){
	case OIS::KC_UP:
	case OIS::KC_W:
		cameraDirection.z = 0;
		break;
	case OIS::KC_DOWN:
	case OIS::KC_S:
		cameraDirection.z = 0;
		break;
	case OIS::KC_LEFT:
	case OIS::KC_A:
		cameraDirection.x = 0;
		break;
	case OIS::KC_RIGHT:
	case OIS::KC_D:
		cameraDirection.x = 0;
		break;
	case OIS::KC_PGDOWN:
	case OIS::KC_E:
		cameraDirection.y = 0;
		break;
	case OIS::KC_PGUP:
	case OIS::KC_Q:
		cameraDirection.y = 0;
		break;
	default:
		break;
	}
	return true;
}

bool MainApplication::mouseMoved(const OIS::MouseEvent &me){
	cameraNode->yaw(Ogre::Degree(-mRotate*me.state.X.rel), Ogre::Node::TS_WORLD);
	cameraNode->pitch(Ogre::Degree(-mRotate*me.state.Y.rel), Ogre::Node::TS_LOCAL);

	return true;
}

bool MainApplication::mousePressed(const OIS::MouseEvent &me, OIS::MouseButtonID id){
	return true;
}

bool MainApplication::mouseReleased(const OIS::MouseEvent &me, OIS::MouseButtonID id){
	return true;
}

#if OGRE_PLATFORM == OGRE_PLATFORM_WIN32
#define WIN32_LEAN_AND_MEAN
#include "windows.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if OGRE_PLATFORM == OGRE_PLATFORM_WIN32
	INT WINAPI WinMain(HINSTANCE hInst, HINSTANCE, LPSTR strCmdLine, INT)
#else
	int main(int argc, char *argv[])
#endif
	{
		string fileName = "TBTKResults.h5";
		if(argc == 2){
			fileName = argv[1];
		}

		// Create application object
		MainApplication app;

		try{
			app.setFileName(fileName);
			app.go();
		}
		catch(Ogre::Exception& e){
#if OGRE_PLATFORM == OGRE_PLATFORM_WIN32
			MessageBox(NULL, e.getFullDescription().c_str(), "An exception has occurred!", MB_OK | MB_ICONERROR | MB_TASKMODAL);
#else
			std::cerr << "An exception has occurred: " <<
			e.getFullDescription().c_str() << std::endl;
#endif
		}

		return 0;
	}
#ifdef __cplusplus
}
#endif
