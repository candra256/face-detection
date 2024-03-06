import cv2
import sys

# dua perpustakaan 1. opencv 2. sys

smile_cascade = cv2.CascadeClassifier("smile_rev.xml")
# identifikasi senyuman dengan file "smile_rev.xml"
camera = cv2.VideoCapture(0)

def smile_detection(frame):
     
     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     smiles = smile_cascade.detectMultiScale(gray_frame, scaleFactor=1.8, minSize=(30, 70),  minNeighbors=25)
     return smiles


def draw_boxes(frame):
     for x, y, w, h in smile_detection(frame):
          cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 4)
          cv2.putText(frame, "Dia tersenyum", (x , y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)


def close_window():
     camera.release()
     cv2.destroyAllWindows()
     sys.exit()

def main():
     while True:
          ret, frame = camera.read()

          if not ret:
               print("gagal membaca frame dari kamera")
               break

          draw_boxes(frame)
          cv2.imshow("smile_detection", frame)
          if cv2.waitKey(1) & 0xFF == ord("q"):
               close_window()

if __name__ == "__main__":
     try:
          main()

     finally:
          close_window()
