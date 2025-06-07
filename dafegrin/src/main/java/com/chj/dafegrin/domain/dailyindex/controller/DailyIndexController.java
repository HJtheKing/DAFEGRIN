package com.chj.dafegrin.domain.dailyindex.controller;

import com.chj.dafegrin.domain.dailyindex.repository.DailyIndexRepository;
import com.chj.dafegrin.entity.DailyIndex;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@Slf4j
@RestController
@RequiredArgsConstructor
@RequestMapping("/dfg/idx")
public class DailyIndexController {
    private final DailyIndexRepository dailyIndexRepository;

    @GetMapping
    public ResponseEntity<?> getDailyIndex() {
        List<DailyIndex> all = dailyIndexRepository.findAll();
        return new ResponseEntity<>(all, HttpStatus.OK);
    }
}
