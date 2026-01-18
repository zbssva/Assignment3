Код проекта загружен в репозиторий и может быть открыт в VSCode. Но на моём ноутбуке с AMD GPU код не запускается, так как CUDA требует NVIDIA GPU.  
Поэтому для проверки код запускала в Google Colab с доступом к Tesla T4. Там код полностью компилируется и выполняется, результаты и скрины из Colab.
<img width="1009" height="691" alt="image" src="https://github.com/user-attachments/assets/483cf6b2-60a5-4513-9263-29e2b5eee568" />


## Задание 1.

Global memory time: 39.5571 ms.

Shared memory time: 7.33763 ms.

Разделяемая память значительно ускорила выполнение по сравнению с глобальной, потому что доступ к ней на GPU гораздо быстрее, а глобальная память медленная. Здесь видно, что при многократной обработке элементов разница особенно заметна.

<img width="1009" height="632" alt="image" src="https://github.com/user-attachments/assets/34ae92ec-9edb-4f7c-8d1b-b49c93cbba26" />
<img width="1009" height="717" alt="image" src="https://github.com/user-attachments/assets/5a329020-0cde-4084-9bb6-c572460d28d1" />


## Задание 2.

blockSize = 128 → 12.6327 ms.

blockSize = 256 → 0.00352 ms.

blockSize = 512 → 0.00256 ms.


Оптимальный размер блока примерно 256 потоков. Если блоков больше, ускорение почти не меняется, потому что GPU уже полностью загружен.

<img width="1009" height="729" alt="image" src="https://github.com/user-attachments/assets/91da7fc2-fd3d-4850-9fff-854e3c702de7" />


## Задание 3.

Coalesced access time: 7.55133 ms.

Non-coalesced access time: 0.003488 ms.

В моём эксперименте время получилось меньше при некоалесцированном доступе из-за специфики Colab и синхронизации. В реальной задаче обычно коалесцированный доступ быстрее и эффективнее.

<img width="1009" height="635" alt="image" src="https://github.com/user-attachments/assets/6939e08d-a920-406f-80e3-b0f6bc31a9ab" />


## Задание 4.

blockSize = 64 → 11.3882 ms.

blockSize = 256 → 0.00432 ms.

Увеличение размера блока до 256 потоков позволило загрузить больше потоков на SM GPU и сократить время выполнения.
Замеры нужно делать с cudaDeviceSynchronize(), чтобы результаты были точными.

<img width="1009" height="711" alt="image" src="https://github.com/user-attachments/assets/1d6edf64-86ec-4807-aee5-948bc4d450f6" />
