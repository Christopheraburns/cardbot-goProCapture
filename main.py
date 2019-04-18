from goprocam import GoProCamera, constants


gpCam = GoProCamera.GoPro()

TIMER=0.1

gpCam.downloadLastMedia(gpCam.take_photo(TIMER), "cardshot.jpg")



