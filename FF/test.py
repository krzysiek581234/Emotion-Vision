from tensorboardX import SummaryWriter

log_dir = 'D:/Studia_Krzysiek/Semestr4/Projekty/Projekt_Sztuczna/Emotion-Vision/logs/FF'  # Ścieżka do katalogu, gdzie chcesz zapisać logi

# Tworzenie instancji SummaryWriter
writer = SummaryWriter(log_dir)

# Przebieg 1
writer.add_scalar('loss', 0.5, global_step=1)
writer.add_scalar('accuracy', 0.9, global_step=1)

# Przebieg 2
writer.add_scalar('loss', 0.3, global_step=2)
writer.add_scalar('accuracy', 0.95, global_step=2)

# Przebieg 3
writer.add_scalar('loss', 0.1, global_step=3)
writer.add_scalar('accuracy', 0.98, global_step=3)

# Możesz kontynuować dodawanie kolejnych przebiegów

# Nie ma potrzeby zamykania SummaryWriter

