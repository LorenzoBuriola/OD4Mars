program pack
    implicit none
    integer, parameter :: nlayermax=55
    integer, parameter :: nspecies=6
    integer, parameter :: nnmax=100000000
    integer degree
    integer npn(nlayermax)
    character(len=100) fin(nspecies)
    character(len=5) speciesname(nspecies)
    character(len=2) specienumber,igstr,degstr
    character(len=2) layernum(nlayermax)
    character(len=10) name_database
    data speciesname/'CO2 ','CO  ' ,'H2O ', 'O3  ', 'HCl ', 'HDO '/
    integer ncharspecies(nspecies)
    data ncharspecies/3,2,3,2,3,3/
    integer equad(nspecies,nlayermax),squad(nspecies,nlayermax)
    integer quad(nnmax)
    integer iw,is,ig,layer,ind,nquad,nnquad
    real, allocatable :: cq(:,:)
    real cc,vind

    call get_command_argument(1, name_database)
    call get_command_argument(2, degstr)
    read(degstr,*) degree
    equad=0
    squad=0
    nnquad=0
    allocate(cq(nnmax,degree+1))

    write(*,*) 'Starting PACK'
    do layer=nlayermax,1,-1
        write(layernum(layer),'(I2.2)') int(layer)
        write(*,'(A,I2)') 'reading layer   ', layer
        do is=1,nspecies
            write(specienumber,'(I0.2)') is
            fin(is)='/home/buriola/OD4Mars/NO_BACKUP/data/s4Mars/' &
                &   //trim(name_database)//'/to_pack/c'//specienumber(1:2)//'0'//layernum(layer)
            squad(is,layer)=nnquad+1
            open(90,file=fin(is),form='unformatted',access='stream',status='old')
            read(90) npn(layer)
            read(90) nquad
            do iw=1,nquad
                read(90) ind
                quad(iw+nnquad)=ind
            enddo
            do iw=1,nquad
                read(90) vind
            enddo
            do ig=1,degree+1
                do iw=1,nquad
                    read(90) cc
                    cq(iw+nnquad,ig)=cc
                enddo
            enddo
            nnquad=nnquad+nquad
            if(nnquad.gt.nnmax) stop 'nnquad.gt.nnmax'
            equad(is,layer)=nnquad
            if(equad(is,layer).eq.0) squad(is,layer)=0
            if(equad(is,layer).lt.squad(is,layer)) then
                squad(is,layer)=0
                equad(is,layer)=0
            endif
            write(*,'(A5,I10,I10,I10)') trim(speciesname(is)), squad(is,layer), equad(is,layer), nnquad
        enddo          !OVER SPECIES
    enddo              !OVER LAYERS

    open(7,file='/home/buriola/OD4Mars/NO_BACKUP/data/s4Mars/'&
        &   //trim(name_database)//'/hc',status='unknown')
    write(7,'(I10)') nnquad
    do layer=1,nlayermax
        write(7,'(I6, (12I9))') npn(layer),(squad(is,layer),equad(is,layer),is=1,nspecies)
    enddo
    close(7)
    if(nnquad.gt.0) then
        open(2,file='/home/buriola/OD4Mars/NO_BACKUP/data/s4Mars/'&
            &   //trim(name_database)//'/quad',form='unformatted',status='unknown')
        write(2) (quad(iw),iw=1,nnquad)
        close(2)
        do ig=1,degree+1
            write(igstr,'(I0)')ig-1
            open(2,file='/home/buriola/OD4Mars/NO_BACKUP/data/s4Mars/'&
                &//trim(name_database)//'/cq'//trim(igstr),form='unformatted',status='unknown')
            write(2) (cq(iw,ig),iw=1,nnquad)
            close(2)
        enddo
    endif
end program pack